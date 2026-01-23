from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import requests
from typing import Dict, List, Any, Optional, TypedDict
from mathruler.grader import extract_boxed_content
from tqdm import tqdm
import numpy as np
import time
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

DISPATCHER_URL = "http://localhost:8001/dispatch"

class DispatcherResponse(TypedDict):
    question: str
    majority_answer: Optional[str]
    self_consistency_score: float
    total_samples: int
    all_answers: List[Optional[str]]
    raw_responses: List[Dict[str, Any]]
    tool_calls: List[int]

def _call_self_consistency_dispatcher(question: str) -> DispatcherResponse:
    """Calls the self-consistency dispatcher for a single question."""
    payload = {
        "question": question,
        "n": 10, # Sampling n=5 for efficiency in the reward function
        "max_turns": 5
    }

    if question == "":
        return {
            "question": question,
            "majority_answer": None,
            "self_consistency_score": 0.0,
            "total_samples": 0,
            "all_answers": [],
            "raw_responses": [],
            "tool_calls": []
        }

    try:
        response = requests.post(DISPATCHER_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling dispatcher: {e}")
        return {
            "question": question,
            "majority_answer": None,
            "self_consistency_score": 0.0,
            "total_samples": 0,
            "all_answers": [],
            "raw_responses": [],
            "tool_calls": []
        }

def reward_self_consistency_scores(questions: List[str], max_threads: int = 5) -> List[DispatcherResponse]:
    """Fetches self-consistency scores for a batch of questions using multiple threads."""
    results = [None] * len(questions)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Map indices to questions to maintain order
        future_to_idx = {executor.submit(_call_self_consistency_dispatcher, q): i for i, q in enumerate(questions)}
        
        for future in tqdm(as_completed(future_to_idx), total=len(questions), desc=" - Requesting self-consistency"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Thread error for index {idx}: {e}")
                results[idx] = {"question": questions[idx], "self_consistency_score": 0.0, "majority_answer": None}
    return results

def reward_format(predict: str):
    # Check for basic <question> and \boxed{ existence first
    pattern_basic = re.compile(r".*<question>.*</question>.*\\boxed\{.*", re.DOTALL)
    if not re.fullmatch(pattern_basic, predict):
        return False
    
    # Robust check for balanced \boxed{...} or use mathruler
    try:
        boxed_content = extract_boxed_content(predict)
        if boxed_content is not None and boxed_content != "None" and boxed_content.strip() != "":
            return True
        else:
            return False
    except Exception:
        return False

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in tqdm(range(n), desc="  - Calculating BLEU distance matrix", leave=False):
        for j in range(i, n):
            if i == j or sentences[i] == sentences[j]:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def penalty_cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def load_historical_questions(directory="questions"):
    historical_questions = []
    if not os.path.exists(directory):
        return []
    
    files = [f for f in os.listdir(directory) if f.startswith("results_sc_") and f.endswith(".json")]
    
    for f in files:
        path = os.path.join(directory, f)
        try:
            with open(path, "r") as fd:
                data = json.load(fd)
                for item in data:
                    sc = item.get("self_consistency_score", 0.0)
                    # only keep questions with sc >= 0.8 or sc <= 0.2
                    if sc >= 0.8 or sc <= 0.2:
                        q = item.get("question", "")
                        if q and isinstance(q, str):
                            historical_questions.append(q)
        except Exception:
            pass
            
    return historical_questions

"""
    Input: predicts is a list of questions
    Output: scores is a list of scores for each question in the same order
"""
def compute_score(predicts: List[str]) -> List[Dict[str, float]]:
    """
    Computes the final reward for curriculum tasks.
    Formula: R = R_format * R_uncertainty * (1 - novelty_penalty)
    """
    print(f"🎯🎯🎯 compute scores for {len(predicts)} predictions")
    results_parsing = []
    lambda_uncertain = 2
    lambda_repetition = 1
    lambda_tool = 0.05
    # 1. Parse predictions to extract questions
    for i in tqdm(range(len(predicts)), desc=" - Parsing predictions"):
        questions = re.findall(r"<question>(.*?)</question>", predicts[i], re.DOTALL)
        if questions:
            results_parsing.append({"question": questions[-1].strip(), "raw_response": predicts[i]})
        else:
            results_parsing.append({"question": "", "raw_response": predicts[i]})

    questions_list = [r["question"] for r in results_parsing]
    
    # 2. Calculate Novelty/Repetition Penalty (BEFORE SC scores)
    # Load historical questions (solved or failed hard)
    historical_questions = load_historical_questions()
    print(f"🎯🎯🎯 Loaded {len(historical_questions)} historical questions (SC>=0.8 or SC<=0.2)")
    
    # Identify which questions are too similar to historical ones
    # Use high threshold (0.8) to only filter very similar questions
    if historical_questions:
        combined = questions_list + historical_questions
        all_proportions = penalty_cluster_share_per_problem(combined, distance_threshold=0.8)
        
        # Slice to get proportions for current questions
        historical_penalties = all_proportions[:len(questions_list)]
        
        # Filter: if a question has high penalty (>0.5), it's likely a duplicate
        # We'll skip SC calculation for these
        novel_indices = []
        filtered_indices = []
        
        for i, penalty in enumerate(historical_penalties):
            if penalty > 0.5:  # High penalty = likely duplicate
                filtered_indices.append(i)
            else:
                novel_indices.append(i)
        
        print(f"🎯🎯🎯 Filtered {len(filtered_indices)} similar questions(sc <= 0.1 or sc >= 0.9), computing SC only for {len(novel_indices)} novel ones")
        
        # Only compute SC for novel questions
        novel_questions = [questions_list[i] for i in novel_indices]
        novel_sc_results = reward_self_consistency_scores(novel_questions, max_threads=5) if novel_questions else []
        
        # Build full sc_results list with placeholders for filtered questions
        sc_results = []
        novel_idx = 0
        for i in range(len(questions_list)):
            if i in filtered_indices:
                # Filtered question: assign dummy low SC result
                sc_results.append({
                    "question": questions_list[i],
                    "majority_answer": None,
                    "self_consistency_score": 0.0,
                    "total_samples": 0,
                    "all_answers": [],
                    "tool_calls": []
                })
            else:
                sc_results.append(novel_sc_results[novel_idx])
                novel_idx += 1
    else:
        # No historical questions, compute SC for all
        historical_penalties = [0.0] * len(questions_list)
        sc_results = reward_self_consistency_scores(questions_list, max_threads=5)
    
    print(f"🎯🎯🎯 Computed {len([r for r in sc_results if r.get('total_samples', 0) > 0])} SC results")

    # Calculate repetition penalty within current batch
    novelty_proportions = penalty_cluster_share_per_problem(questions_list)
    print(f"🎯🎯🎯 Computed {len(novelty_proportions)} Repetition results")

    import json
    # Filter results for JSON logging - exclude large raw_responses
    sc_logs = [{
        "question": res.get("question", ""),
        "majority_answer": res.get("majority_answer"),
        "self_consistency_score": res.get("self_consistency_score", 0.0),
        "total_samples": res.get("total_samples", 0),
        "all_answers": res.get("all_answers", []),
        "tool_calls": res.get("tool_calls", []),
        "pre_filtered": res.get("total_samples", 0) <= 0 and historical_questions
    } for res in sc_results]

    os.makedirs("questions", exist_ok=True)
    with open(f'questions/results_sc_{int(time.time())}.json', 'w') as f:
        json.dump(sc_logs, f, indent=2)

    # 3. Get Novelty Penalty (Clustering)
    # We only cluster valid (non-empty) questions to avoid penalization for empty strings

    final_scores = []
    for i in range(len(predicts)):
        # A. Format Reward (0 or 1)
        fmt_reward = reward_format(predicts[i])
        
        # Check if this question was filtered (skipped SC calculation)
        sc_res = sc_results[i]
        was_filtered = sc_res.get("total_samples", 0) == 0 and historical_questions
        
        if was_filtered:
            # Filtered question: use simplified reward based on historical penalty
            # overall = max(0, 1 - historical_penalty)
            historical_penalty = historical_penalties[i] if historical_questions else 0.0
            overall_reward = max(0, 1 - historical_penalty) * lambda_repetition if fmt_reward else 0
            
            final_scores.append({
                "overall": float(overall_reward),
                "format_score": 1 if fmt_reward is True else 0,
                "uncertainty_score": 0.0,
                "repetition_penalty": float(historical_penalty),
                "sc_score": 0.0,
                "avg_tool_calls": 0.0
            })
        else:
            # Novel question: use full reward calculation
            # B. Uncertainty Reward (Tent Function)
            # f(p) = 1 - |2p - 1| 
            # Peaks at p=0.5 (reward=1.0), drops to 0 at p=0 and p=1.
            p_x = sc_res.get("self_consistency_score", 0.0)
            # it has been normalized between 0.0 and 1.0
            uncertainty_reward = 1.0 - 2 * abs(p_x - 0.5) # uncertity is 0.5
            repetition_penalty = novelty_proportions[i]
            
            # Calculate tool calls mean
            tool_counts = sc_res.get("tool_calls", [])
            avg_tool_calls = min(np.mean(tool_counts) if tool_counts else 0.0, 4)
            
            # Combine
            # R = Rformat(xi) · max(0,λuncRunc + λtoolRtool − Rrep(xi))
            overall_reward = max(0, lambda_uncertain * uncertainty_reward + lambda_tool * avg_tool_calls - lambda_repetition * repetition_penalty) if fmt_reward else 0

            final_scores.append({
                "overall": float(overall_reward),
                "format_score": 1 if fmt_reward is True else 0,
                "uncertainty_score":  lambda_uncertain * float(uncertainty_reward),
                "repetition_penalty": lambda_repetition * float(repetition_penalty),
                "sc_score": float(p_x),
                "avg_tool_calls": float(avg_tool_calls)
            })

    return final_scores
