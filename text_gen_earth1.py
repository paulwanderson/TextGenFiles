from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def run():
	tokenizer = AutoTokenizer.from_pretrained("gpt2")
	model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

	print("Model Loaded..!")

	start_time = time.time()

	#Change input text to be the stem of the misinfo article/paragraph
	input_text = "The Earth is a flat disk centered at the North Pole with boundaries at its edges made of enormous walls of ice, with the sun, moon, stars and planets a few hundred miles above us in the sky."
	inputs = tokenizer(input_text, return_tensors="pt")
	input_ids = inputs["input_ids"]

	output = model.generate(
	input_ids,
	attention_mask=inputs["attention_mask"],
	do_sample=True,
	# Change max_length to be article length
	max_length=250,
	temperature=0.8,
	use_cache=True,
	top_p=0.9
	)

	end_time = time.time() - start_time
	print("Total Taken => ",end_time)
	print(tokenizer.decode(output[0]))