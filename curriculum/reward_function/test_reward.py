import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from curriculum.reward_function.curriculum import compute_score

def test_reward_logic():
    print("Starting reward logic verification...")
    
    # Mocking the dispatcher to return specific self-consistency scores
    # 1. format error (missing boxed)
    # 2. perfect uncertainty (p=0.5)
    # 3. low uncertainty (p=1.0)
    # 4. redundant question (will be tested via clustering)
    
    predicts = [
        "<question>What is 1+1?</question> 1+1 is 2.", # Bad format (no boxed)
        "<question>Find the root of x^2-4=0</question> \\boxed{2}", # Perfect uncertainty (mock p=0.5)
        "<question>What is prime?</question> \\boxed{7}", # Low uncertainty (mock p=1.0)
        "<question>What is 2+2?</question> \\boxed{4}", # Duplicate
        "<question>What is 2+2?</question> \\boxed{4}", # Duplicate
    ]
    
    # Mock sc_results
    mock_sc_results = [
        {"self_consistency_score": 0.5, "question": "What is 1+1?"},
        {"self_consistency_score": 0.5, "question": "Find the root of x^2-4=0"},
        {"self_consistency_score": 1.0, "question": "What is prime?"},
        {"self_consistency_score": 0.8, "question": "What is 2+2?"},
        {"self_consistency_score": 0.8, "question": "What is 2+2?"},
    ]

    with patch('curriculum.reward_function.curriculum.reward_self_consistency_scores') as mock_sc:
        mock_sc.return_value = mock_sc_results
        
        scores = compute_score(predicts)
        
        for i, score in enumerate(scores):
            print(f"\nPrediction {i}:")
            print(f"  Overall: {score['overall']:.4f}")
            print(f"  Format: {score['format_score']}")
            print(f"  Uncertainty: {score['uncertainty_score']}")
            print(f"  Novelty: {score['novelty_score']:.4f}")
            print(f"  SC: {score['sc_score']}")

        # Assertions
        # 0: Format should be 0, so Overall should be 0
        assert scores[0]['format_score'] == 0.0
        assert scores[0]['overall'] == 0.0
        
        # 1: Format 1, Uncertainty (p=0.5) 1.0, Novelty should be high (unique)
        assert scores[1]['format_score'] == 1.0
        assert scores[1]['uncertainty_score'] == 1.0
        assert scores[1]['overall'] > 0
        
        # 2: Format 1, Uncertainty (p=1.0) 0.0
        assert scores[2]['uncertainty_score'] == 0.0
        assert scores[2]['overall'] == 0.0
        
        # 3 & 4: Should have lower novelty score than 1 and 2
        assert scores[3]['novelty_score'] < scores[1]['novelty_score']
        
    print("\nReward logic verification PASSED!")

if __name__ == "__main__":
    try:
        test_reward_logic()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
