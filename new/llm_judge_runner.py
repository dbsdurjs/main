# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runner for LLM Judge - 한국어 지원 + Rating label 원본 순서 변환."""

from collections.abc import Sequence
import math
from typing import Optional

from llm_comparator import _logging
from llm_comparator import model_helper
from llm_comparator import prompt_templates
from llm_comparator import types
from llm_comparator import utils


_IndividualRating = types.IndividualRating
_JsonDict = types.JsonDict
_LLMJudgeInput = types.LLMJudgeInput
_LLMJudgeOutput = types.LLMJudgeOutput
_GenerationModelHelper = model_helper.GenerationModelHelper

_logger = _logging.logger


# 영어 + 한국어 레이블 모두 지원
DEFAULT_RATING_TO_SCORE_MAP = {
    # 영어 레이블
    'A is much better': 1.5,
    'A is better': 1.0,
    'A is slightly better': 0.5,
    'same': 0.0,
    'B is slightly better': -0.5,
    'B is better': -1.0,
    'B is much better': -1.5,
    # 한국어 레이블
    'A가 훨씬 더 좋음': 1.5,
    'A가 더 좋음': 1.0,
    'A가 약간 더 좋음': 0.5,
    '비슷함': 0.0,
    'B가 약간 더 좋음': -0.5,
    'B가 더 좋음': -1.0,
    'B가 훨씬 더 좋음': -1.5,
}


def flip_rating_label(label: str) -> str:
  """Rating label의 A와 B를 뒤집습니다.
  
  Args:
    label: 원본 rating label
    
  Returns:
    A와 B가 뒤집힌 rating label
  """
  # 한국어 레이블
  if label == 'A가 훨씬 더 좋음':
    return 'B가 훨씬 더 좋음'
  elif label == 'A가 더 좋음':
    return 'B가 더 좋음'
  elif label == 'A가 약간 더 좋음':
    return 'B가 약간 더 좋음'
  elif label == 'B가 훨씬 더 좋음':
    return 'A가 훨씬 더 좋음'
  elif label == 'B가 더 좋음':
    return 'A가 더 좋음'
  elif label == 'B가 약간 더 좋음':
    return 'A가 약간 더 좋음'
  elif label == '비슷함':
    return '비슷함'
  
  # 영어 레이블
  elif label == 'A is much better':
    return 'B is much better'
  elif label == 'A is better':
    return 'B is better'
  elif label == 'A is slightly better':
    return 'B is slightly better'
  elif label == 'B is much better':
    return 'A is much better'
  elif label == 'B is better':
    return 'A is better'
  elif label == 'B is slightly better':
    return 'A is slightly better'
  elif label == 'same':
    return 'same'
  
  # 알 수 없는 레이블은 그대로 반환
  return label


class LLMJudgeRunner:
  """Runner for LLM judge that determines which response is better."""

  def __init__(
      self,
      generation_model_helper: _GenerationModelHelper,
      llm_judge_prompt_template: str = prompt_templates.DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
      rating_to_score_map: Optional[dict[str, float]] = None,
  ):
    """Initializes the LLM judge runner.

    Args:
      generation_model_helper: Generative model helper to run the LLM judge.
      llm_judge_prompt_template: Prompt template for LLM judge.
      rating_to_score_map: Map from rating label text to score.
    """
    self.generation_model_helper = generation_model_helper
    self.llm_judge_prompt_template = llm_judge_prompt_template
    if rating_to_score_map is None:
      rating_to_score_map = DEFAULT_RATING_TO_SCORE_MAP
    self.rating_to_score_map = rating_to_score_map

  def create_prompt_for_judge(
      self, prompt: str, response_a: str, response_b: str
  ) -> str:
    prompt_for_judge = self.llm_judge_prompt_template.format(
        prompt=prompt, response_a=response_a, response_b=response_b
    )
    return prompt_for_judge

  def create_inputs_with_repeats_for_judge(
      self, inputs: Sequence[_LLMJudgeInput], num_repeats: int
  ) -> Sequence[_JsonDict]:
    """Creates inputs with repeated runs for LLM Judge."""
    inputs_with_repeats = []
    for index, ex in enumerate(inputs):
      # Non-flipped.
      # If num_repeats is an odd number, roundup.
      for _ in range(math.ceil(num_repeats * 0.5)):
        inputs_with_repeats.append({
            'example_index': index,
            'prompt': ex['prompt'],
            'response_a': ex['response_a'],
            'response_b': ex['response_b'],
            'is_flipped': False,
        })
      # Flipped.
      # If num_repeats is an odd number, rounddown.
      for _ in range(math.floor(num_repeats * 0.5)):
        inputs_with_repeats.append({
            'example_index': index,
            'prompt': ex['prompt'],
            'response_a': ex['response_b'],
            'response_b': ex['response_a'],
            'is_flipped': True,
        })
    return inputs_with_repeats

  def run(
      self,
      inputs: Sequence[_LLMJudgeInput],
      num_repeats: int = 6,
  ) -> Sequence[_LLMJudgeOutput]:
    """Runs LLM judge on the given inputs.

    Args:
      inputs: Inputs for LLM judge.
      num_repeats: Number of times to repeat each input for LLM judge.

    Returns:
      List of LLM judge outputs.
    """
    inputs_for_judge = self.create_inputs_with_repeats_for_judge(
        inputs, num_repeats
    )
    prompts_for_judge = [
        self.create_prompt_for_judge(
            ex['prompt'], ex['response_a'], ex['response_b']
        )
        for ex in inputs_for_judge
    ]
    outputs_from_judge = self.generation_model_helper.predict_batch(
        prompts_for_judge
    )
    example_ratings = self.parse_outputs(inputs_for_judge, outputs_from_judge)
    scores_and_ratings = self.postprocess_results(example_ratings)
    return scores_and_ratings

  def parse_outputs(
      self,
      inputs_for_judge: Sequence[_JsonDict],
      outputs_from_judge: Sequence[str],
  ) -> Sequence[Sequence[_IndividualRating]]:
    """Parses LLM judge outputs."""

    def parse_output(output: str) -> Optional[tuple[float, str, str]]:
      parsed_xml = utils.extract_xml_part(output, 'result')
      if parsed_xml is None:
        return None

      if (rationale := parsed_xml.find('explanation')) is None:
        return None
      if (rationale := rationale.text) is None:
        return None

      if (rating_label := parsed_xml.find('verdict')) is None:
        return None
      if (rating_label := rating_label.text) is None:
        return None

      try:
        score = self.rating_to_score_map[rating_label]
      except KeyError:
        _logger.error(
            'LLM judge returned an unknown rating label: %s}', rating_label
        )
        return None
      return (score, rating_label, rationale.strip(' \n'))

    max_example_index = max([ex['example_index'] for ex in inputs_for_judge])
    example_ratings = [[] for _ in range(max_example_index + 1)]

    for judge_input, raw_output in zip(inputs_for_judge, outputs_from_judge):
      parsed_output = parse_output(raw_output)
      if parsed_output:
        original_score = parsed_output[0]
        original_label = parsed_output[1]
        rationale = parsed_output[2]
        
        # flipped인 경우 score와 label 모두 변환
        if judge_input['is_flipped']:
          converted_score = original_score * -1.0
          converted_label = flip_rating_label(original_label)
        else:
          converted_score = original_score
          converted_label = original_label
        
        example_ratings[judge_input['example_index']].append({
            'is_flipped': judge_input['is_flipped'],
            'score': converted_score,
            'rating_label': converted_label,  # 원본 순서로 변환된 label
            'rationale': rationale,
        })
    _logger.info('Parsed %d example ratings.', len(example_ratings))
    return example_ratings

  def postprocess_results(
      self, example_ratings: Sequence[Sequence[_IndividualRating]]
  ) -> Sequence[_LLMJudgeOutput]:
    results: list[_LLMJudgeOutput] = []
    for ratings in example_ratings:
      # division by zero 방지
      if len(ratings) == 0:
        _logger.warning('No valid ratings found for an example, skipping.')
        score = 0.0
      else:
        score = sum([rating['score'] for rating in ratings]) / len(ratings)
      results.append({
          'score': score,
          'individual_rater_scores': list(ratings),
      })
    return results