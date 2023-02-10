import glob
import logging
import os
import random
import scipy.spatial
from nli import BertNLI
from object_detection import ObjectDetection
from sentence_bert import SentenceBertJapanese
from vqa import Vqa
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


class PersonaCaption:
    def __init__(self):
        self.model = SentenceBertJapanese()

        self.persona_data = {}
        with open("./data/persona_list.csv") as f:
            for i, text in enumerate(f):
                if i == 0:
                    continue
                print(i, "...", end="\r")
                persona = text.split(",")
                desc = persona[1].strip()
                label = persona[2].strip()
                self.persona_data[desc] = label

    def _get_query_list(self, image_path):
        object_detection = ObjectDetection()

        output = object_detection.detection(image_path)
        object_labels = object_detection.get_object_labels(output)
        normalized_boxes, roi_features = object_detection.get_object_features_for_vlt5(
            output
        )

        vqa = Vqa(normalize_boxes=normalized_boxes, roi_features=roi_features)
        questions = []
        with open("./data/vqa_questions.txt") as f:
            for question in f.readlines():
                questions.append(question)
        vqa_answers = vqa.get_answer(questions)

        return object_labels, vqa_answers

    def _get_query_score_dict(self, image_path, output_size=5):
        object_labels, vqa_answers = self._get_query_list(image_path)
        # If there are duplicate query in object labels and vqa answers,
        # remove them from vqa answers.
        for label in object_labels:
            if label in vqa_answers:
                vqa_answers.remove(label)
        # The score of a query obtained by object detection is 1.
        object_labels_score_dict = {l: 1.0 for l in object_labels}
        # The score of a quey obtained by vqa is 0.9.
        vqa_answers_score_dict = {v: 0.9 for v in vqa_answers}
        query_score_dict = {**object_labels_score_dict, **vqa_answers_score_dict}

        if not os.path.exists("./data/chive"):
            logger.error("chiVe vector is not placed under ./data/chive")

        kv_file = glob.glob("./data/chive/*.kv")
        npy_file = glob.glob("./data/chive/*.npy")
        if not kv_file or not npy_file:
            logger.error(
                "chiVe vector .kv or .npy file is not placed under ./data/chive"
            )
            logger.warn("Could not extract synonyms.")
            return query_score_dict

        word2vec_model = KeyedVectors.load(kv_file[0])
        for query in list(query_score_dict.keys()):
            if query in word2vec_model:
                for sim in word2vec_model.most_similar(query, topn=output_size):
                    synonym = sim[0]
                    cos_sim = round(float(sim[1]), 3)
                    # The score of synonym is the product of the cos similarity value and the query score.
                    synonym_score = round(float(cos_sim * query_score_dict[query]), 3)
                    if synonym not in query_score_dict or (
                        synonym in query_score_dict
                        and synonym_score > query_score_dict[synonym]
                    ):
                        query_score_dict[synonym] = synonym_score
        logger.info("Successfully build query score dict. dict = %s", query_score_dict)
        return query_score_dict

    def _search(self, query_score_dict, distance_threshold=1):
        logger.info("Searching...")
        search_queries = list(query_score_dict.keys())
        persona_sentences = list(self.persona_data.keys())
        sentence_vectors = self.model.encode(persona_sentences)
        query_embeddings = self.model.encode(search_queries).numpy()

        search_result = {}
        for query, query_embedding in zip(search_queries, query_embeddings):
            query_score = query_score_dict[query]
            # cos_distance = 1- cos_similarity
            distances = scipy.spatial.distance.cdist(
                [query_embedding], sentence_vectors, metric="cosine"
            )[0]

            results = zip(range(len(distances)), distances)
            # sort by cos_distance ascending
            results = sorted(results, key=lambda x: x[1])
            for idx, distance in results:
                if distance < distance_threshold:
                    persona_sentence = persona_sentences[idx]
                    persona_score = self._get_persona_score(query_score, distance)
                    if persona_sentence not in search_result or (
                        persona_sentence in search_result
                        and persona_score > search_result[persona_sentence]
                    ):
                        # Add score to each persona.
                        search_result[persona_sentence] = persona_score
        search_result = sorted(search_result.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            "Successfully search by queries. Top 30 search result = %s",
            search_result[:30],
        )
        return search_result

    def _get_persona_score(self, query_score, distance):
        return query_score / (distance + 1)

    def get_persona_list(self, image_path, persona_output_num):
        query_score_dict = self._get_query_score_dict(image_path)
        search_result = self._search(query_score_dict)
        persona_list = []
        label_result = []

        for result in search_result:
            new_persona = result[0]
            # Skip if the persona category duplicates or contradicts any of the previous personas.
            label = self.persona_data[new_persona]
            if (label != "その他" and label in label_result) or self._is_contradiction(
                persona_list, new_persona
            ):
                logger.info(
                    "Persona 「%s」 were skipped without adding to persona list.",
                    new_persona,
                )
                continue
            persona_list.append(new_persona)
            label_result.append(label)
            if len(persona_list) >= persona_output_num:
                break

        logger.info(
            "Successfully get persona caption. persona caption = %s", persona_list
        )
        return persona_list

    def _is_contradiction(self, persona_list, new_persona):
        nli = BertNLI()
        for persona in persona_list:
            if nli.predict(persona, new_persona) == "contradiction":
                logger.info(
                    "Persona 「%s」 contradicts 「%s」 in persona list",
                    new_persona,
                    persona,
                )
                return True
        return False

    def get_random_persona_list(self, persona_output_num):
        return random.sample(list(self.persona_data.keys()), persona_output_num)
