from transformers import AutoTokenizer, GPTJForCausalLM
import time
 
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to("cuda")
 
print("Model Loaded..!")
 
start_time = time.time()
 
input_text = "The Earth is a flat disk centered at the North Pole with boundaries at its edges made of enormous walls of ice, with the sun, moon, stars and planets a few hundred miles above us in the sky."
 
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
      
output = model.generate(
   input_ids,
   attention_mask=inputs["attention_mask"].to("cuda"),
   do_sample=True,
   max_length=2000,
   temperature=0.8,
   use_cache=True,
   top_p=0.9
)
 
end_time = time.time() - start_time
print("Total Time => ",end_time)
print(tokenizer.decode(output[0]))