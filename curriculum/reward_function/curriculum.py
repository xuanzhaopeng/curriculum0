from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import Dict, List
from mathruler.grader import extract_boxed_content

from tqdm import tqdm


def compute_socre(predicts: List[str], ground_truhts: List[str]) -> List[Dict[str,float]]:
    results = []
    with open('test.json','w') as f:
        json.dump(predicts,f,indent=4)

    # extract the question and predicted answer
    for i in tqdm(range(len(predicts), desc=" - Parsing predictions")):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        answers = extract_boxed_content(predicts[i])
        if questions and answers:
            try:
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results.append({"question": question, "answer": answer})
            except:
                results.append({"question": "", "answer": ""})
        else:
            results.append({"question": "", "answer": ""})

    print(f" - {len(results)} results parsed.")

    # Forward pass from agents, and collect result
    # final_results = generate_results(results)
    final_results = results
    scores = []
    for i in tqdm(range(len(final_results)), desc=" - Calculating final scores"):
        scores.append({
            # in RewardManager, the "overall" will be used for reward_tensor, 
            # and other field will be used for reward_metrics
            "overall": 1 if final_results[i]['question'] and final_results[i]['answer'] else 0
        })
    return scores
