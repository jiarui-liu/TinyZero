import re
import random
import json


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    solution_str = solution_str.split("\n")[-1]

    # Extract answer between tags
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip().lower()
        final_answer = final_answer.rstrip(".")
    else:
        final_answer = None
    return final_answer


def validate_answer(answer, question_type):
    """Validate that the answer is in the correct format (yes/no)."""
    if question_type in [
        "tom:answerability:binary",
        "tom:info_accessibility:binary",
    ] and answer in ["yes", "no"]:
        return True

    if question_type in [
        "tom:belief:accessible:multiple-choice",
        "tom:belief:inaccessible:multiple-choice",
    ] and answer in ["(a)", "(b)"]:
        return True

    return False


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """The scoring function for FANTOM tasks.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing the correct answer
        method: the method to extract the solution (not used currently)
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    correct_answer = ground_truth["answer"].lower()

    # Extract the answer from the solution
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Correct answer: {correct_answer}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        if do_print:
            print(f"No answer found")
        return 0

    # Validate answer format
    if not validate_answer(answer, ground_truth["question_type"]):
        if do_print:
            print(f"Invalid answer format")
        return format_score

    # Check if answer is correct
    if correct_answer == answer:
        if do_print:
            print(f"Correct answer: {answer}")
        return score
    else:
        if do_print:
            print(f"Wrong answer: got {answer}, expected {correct_answer}")
        return format_score
