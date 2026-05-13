import torch
from datasets import load_dataset


def load_rwku_datasets():
    forget_data = load_dataset("jinzhuoran/RWKU", 'forget_level1', split='test')
    neighbor_data = load_dataset("jinzhuoran/RWKU", 'neighbor_level1', split='test')
    train_data = load_dataset("jinzhuoran/RWKU", 'train_original_passage', split='train')
    
    return forget_data, neighbor_data, train_data

def load_rwku_data(subject: str):
    forget_data, neighbor_data, train_data = load_rwku_datasets()

    person_forget = forget_data.filter(lambda x: subject in x['subject'])
    person_neighbor = neighbor_data.filter(lambda x: subject in x['subject'])
    person_train = train_data.filter(lambda x: subject in x['subject'])

    print(f"Questions for forgetting test: {len(person_forget)}")
    print(f"Questions for general knowledge test: {len(person_neighbor)}")
    print(f"Texts for training (unlearning): {len(person_train)}")

    return (
        person_train,
        person_forget['query'],
        person_forget['answer'],
        person_neighbor['query'],
        person_neighbor['answer'],
    )


def prepare_tokenized_dataset(person_train, tokenizer):
    forget_dataset = person_train.map(lambda example: {"text_formatted": example['text']})

    def tokenize_function(examples):
        return tokenizer(examples["text_formatted"], padding="max_length", truncation=True, max_length=200)

    tokenized = forget_dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.map(lambda x: {'labels': x['input_ids']})
    print("Data is ready")
    return tokenized


def evaluate_neighbours(model, tokenizer, questions, expected_keywords, device):
    print("Evaluating neighbour (general) knowledge\n")
    return evaluate_model(model, tokenizer, questions, expected_keywords, device)


ANSWER_ALIASES = {
    "joe biden": ["biden"],
    "hillary clinton": ["hillary", "clinton"],
    "amy coney barrett": ["amy", "barrett"],
    "kamala harris": ["kamala", "harris"],
    "donald trump": ["trump", "donald"],
    "barack obama": ["obama", "barack"],
    "muslim-majority": ["muslim"],

}


def keywords_match(keyword, generated_text):
    if keyword in generated_text:
        return True

    aliases = ANSWER_ALIASES.get(keyword, [])
    for alias in aliases:
        if alias in generated_text:
            return True

    return False


def evaluate_model(model, tokenizer, questions, expected_keywords, device):
    model.eval()
    correct_answers = 0
    total_questions = len(questions)

    print("Starting model testing\n")
    print("-" * 50)

    for i in range(total_questions):
        question = questions[i]
        keyword = expected_keywords[i].lower()

        prompt = (f"Task: Fill in the blank (___) in the sentence below. Output ONLY the answer and no further explanation."
                  f" \nSentence: {question}\nanswer:")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response.replace(prompt, "").strip().lower()
        hit = keywords_match(keyword, generated_text)

        if hit:
            correct_answers += 1
            status = "PASSED"
        else:
            status = "FAILED (or forgot)"

        print(f"Question: {question}")
        print(f"Expected: '{keyword}'")
        print(f"Model generated: '{generated_text}'")
        print(f"Result: {status}\n")
        print("-" * 50)

    accuracy = (correct_answers / total_questions) * 100
    print(f"\nFinal Accuracy result: {accuracy:.2f}%")
    return accuracy


def check_dataset_structure(forget_data = None, neighbor_data = None, train_data = None):

    if forget_data and neighbor_data and train_data is None:
        forget_data, neighbor_data, train_data = load_rwku_datasets()

    print("Dataset Structure")
    print("Columns in forget_data:", forget_data.column_names)
    print("Columns in train_data:", train_data.column_names)
    print("Columns in neighbor_data:", neighbor_data.column_names)

    print("Example Rows")
    print("Example row in forget_data", forget_data[0])
    print("Example row in neighbor_data:", neighbor_data[0])

    column_name = 'subject'

    if column_name in forget_data.column_names:
        available_people = list(set(forget_data[column_name]))
        print(f"Found {len(available_people)} people")

        for person in available_people:
            print(f"- {person}")
        else:
            print(f"\nError: Column '{column_name}' is not in the dataset. Update the 'column_name' based on 'Dataset Structure' output.")

    print("Columns in neighbor:", neighbor_data.column_names)
    print("Example row in neighbor:", neighbor_data[0])