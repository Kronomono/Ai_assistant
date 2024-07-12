from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__":
    # Path to your model
    model_path = "minstral"

    try:
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # This might be necessary for some custom models
        )

        # Define your prompt
        prompt = "You are an AI assistant I created to help me with my tasks."

        # Tokenize the input and generate output
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=20)

        # Decode and print the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated output:")
        print(generated_text)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure the model files exist and the path is correct.")