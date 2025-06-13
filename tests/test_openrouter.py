# # We don't run this by default because it uses API calls.

# import asyncio
# from dataclasses import dataclass
# from mypkg.pipeline.infra.model_inference import run_model_inference_openrouter


# @dataclass
# class PromptResult:
#     prompt: str
#     response: str = ""


# async def test_prompt_responses():
#     # Number of prompts you want to test.
#     num_prompts = 50
#     prompt_results = []

#     # Create test prompts that ask the LLM to return the number.
#     for i in range(num_prompts):
#         prompt_text = f"Please return the number {i} as the only part of your response."
#         prompt_results.append(PromptResult(prompt=prompt_text))

#     # Set your model name. Adjust this to your desired API model.
#     model_name = "google/gemma-2-27b-it"

#     # Call your function that sends all prompts to the API.
#     results = await run_model_inference_openrouter(prompt_results, model_name)

#     # Validate that each response exactly equals the expected number.
#     for i, result in enumerate(results):
#         expected = str(i)
#         # You might want to perform a stricter check depending on the model's behavior.
#         if result.response.strip() == expected:
#             print(f"Prompt {i}: PASS (Response: '{result.response.strip()}')")
#         else:
#             print(
#                 f"Prompt {i}: FAIL - Expected: '{expected}', Got: '{result.response.strip()}'"
#             )


# if __name__ == "__main__":
#     asyncio.run(test_prompt_responses())
