from vllm import LLM, SamplingParams


model_name = "facebook/opt-125m"
prompt = "The future of artificial intelligence is"

print("Model:", model_name)
print("Prompt:", prompt)
print("Creating the vLLM runtime...")

llm = LLM(
    model=model_name,
    dtype="float32",
    max_model_len=128,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=16,
)

print("Generating...")
outputs = llm.generate(
    [prompt],
    sampling_params,
)

print("\n**********")
print(outputs[0])
print(outputs[0].outputs[0])
print(outputs[0].outputs[0].text)
print(outputs[0].outputs[0].token_ids)
print("**********")
