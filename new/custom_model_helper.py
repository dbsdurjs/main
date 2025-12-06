#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom Model Helper - MAX_TOKENS 문제 해결
기본 256 토큰 → 2048 -> 8192토큰
"""

import time
from typing import Optional, Sequence
from vertexai import generative_models
from vertexai import language_models
import tqdm.auto


MAX_NUM_RETRIES = 5
DEFAULT_MAX_OUTPUT_TOKENS = 8192  # ✅ 256 → 8192 증가
BATCH_EMBED_SIZE = 100


class VertexGenerationModelHelper:
    """Vertex AI text generation model API calls with increased token limit."""

    def __init__(self, model_name='gemini-pro'):
        self.engine = generative_models.GenerativeModel(model_name)

    def predict(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> str:
        if not prompt:
            return ''
        num_attempts = 0
        response = None
        prediction = None

        while num_attempts < MAX_NUM_RETRIES and response is None:
            num_attempts += 1

            try:
                prediction = self.engine.generate_content(
                    prompt,
                    generation_config=generative_models.GenerationConfig(
                        temperature=temperature,
                        candidate_count=1,
                        max_output_tokens=max_output_tokens,
                    ),
                )
            except Exception as e:
                if 'quota' in str(e):
                    print('\033[31mQuota limit exceeded.\033[0m')
                wait_time = 2**num_attempts
                print(f'\033[31mWaiting {wait_time}s to retry...\033[0m')
                time.sleep(2**num_attempts)

        if prediction is None:
            return ''

        try:
            return prediction.text
        except Exception as e:
            # MAX_TOKENS 오류 처리
            print(f'\033[33mWarning: Cannot get text from prediction: {e}\033[0m')
            
            # candidates 확인
            if hasattr(prediction, 'candidates') and prediction.candidates:
                candidate = prediction.candidates[0]
                
                # finish_reason 확인
                if hasattr(candidate, 'finish_reason'):
                    print(f'\033[33mFinish reason: {candidate.finish_reason}\033[0m')
                
                # content 확인
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    if candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, 'text'):
                            return part.text
            
            return ''

    def predict_batch(
        self,
        prompts: Sequence[str],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> Sequence[str]:
        outputs = []
        for i in tqdm.auto.tqdm(range(0, len(prompts))):
            outputs.append(self.predict(prompts[i], temperature, max_output_tokens))
        return outputs


class VertexEmbeddingModelHelper:
    """Vertex AI text embedding model API calls."""

    def __init__(self, model_name: str = 'text-embedding-004'):
        self.model = language_models.TextEmbeddingModel.from_pretrained(model_name)

    def _embed_single_run(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        """Embeds a list of strings into the models embedding space."""
        num_attempts = 0
        embeddings = None

        if not isinstance(texts, list):
            texts = list(texts)

        while num_attempts < MAX_NUM_RETRIES and embeddings is None:
            try:
                embeddings = self.model.get_embeddings(texts)
            except Exception as e:
                wait_time = 2**num_attempts
                print(f'Waiting {wait_time}s to retry... ({e})')
                time.sleep(wait_time)
                num_attempts += 1

        if embeddings is None:
            return []

        return [embedding.values for embedding in embeddings]

    def embed(self, text: str) -> Sequence[float]:
        results = self._embed_single_run([text])
        return results[0] if results else []

    def embed_batch(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if len(texts) <= BATCH_EMBED_SIZE:
            return self._embed_single_run(texts)
        else:
            results = []
            for batch_start_index in tqdm.auto.tqdm(
                range(0, len(texts), BATCH_EMBED_SIZE)
            ):
                results.extend(
                    self._embed_single_run(
                        texts[batch_start_index : batch_start_index + BATCH_EMBED_SIZE]
                    )
                )
            return results

