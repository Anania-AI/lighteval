from lighteval.tasks.lighteval_task import LightevalTaskConfig, Doc
from lighteval.metrics.metrics import Metrics
from .text_tagging_metric import text_tagging_metric

prompt_language = "en"
SIB200_LABEL_MAP = {
    "technology": "տեխնոլոգիա",
    "travel": "ճանապարհորդություն",
    "business": "բիզնես",
    "disasters": "աղետներ",
    "sports": "սպորտ",
    "science": "գիտություն",
    "nature": "բնություն",
    "politics": "քաղաքականություն",
    "religion": "կրոն",
    "education": "կրթություն",
    "crime": "հանցագործություն",
    "entertainment": "ժամանց",
    "geography": "աշխարհագրություն",
    "health": "առողջապահություն",
}
SIB200_LABELS = list(SIB200_LABEL_MAP.values())
SENTIMENT_LABELS = ["դրական", "բացասական", "չեզոք", "երկիմաստ"]
URGENCY_LABEL_MAP = {"high": "բարձր", "medium": "միջին", "low": "ցածր"}
URGENCY_LABELS = list(URGENCY_LABEL_MAP.values())

PROMPTS_SIB200 = {
    "instruction": {
        "hy": (
            "Տրված տեքստի համար ընտրիր այն թեման, որը լավագույնս նկարագրում է այն։\n"
        ),
        "en": ("For the given text, choose the topic that best describes it.\n"),
    },
    "query": {
        "hy": ("Տեքստ\n{text}\n\n" "Հնարավոր թեմաներ՝ {labels}"),
        "en": ("Text: {text}\n" "Possible topics: {labels}"),
    },
}

PROMPTS_SENTIMENT = {
    "instruction": {
        "hy": ("Որոշիր տեքստի սենտիմենտը։\n"),
        "en": ("Determine the sentiment of the text.\n"),
    },
    "query": {
        "hy": (
            "Տեքստ\n{text}\n\n"
            "Հնարավոր տարբերակներ՝ դրական, բացասական, չեզոք, երկիմաստ"
        ),
        "en": (
            "Text: {text}\n" "Possible options: positive, negative, neutral, ambiguous."
        ),
    },
}
PROMPTS_URGENCY = {
    "instruction": {
        "hy": ("Որոշիր նամակի կարևորության մակարդակը։\n"),
        "en": ("Determine the urgency level of the email.\n"),
    },
    "query": {
        "hy": ("Նամակ\n{text}\n\n" "Տարբերակներ՝ բարձր, միջին, ցածր"),
        "en": ("Email: {text}\n" "Options: high, medium, low"),
    },
}

PROMPTS_TEXT_TAGGING = {
    "instruction": {
        "hy": (
            "Տրված տեքստից առանձնացրեք և գրեք բանալի բառեր ու բառակապակցություններ։ Գրեք արդյունքը` բաժանված ստորակետերով։\n"
        ),
        "en": (
            "Extract and write key words and phrases from the given text. Write the result as a single sentence, separated by commas.\n"
        ),
    },
    "query": {"hy": ("{text}"), "en": ("{text}")},
}


def prompt_sib200(line, task_name=None):
    eng_label = line["category"]
    gold = SIB200_LABEL_MAP[eng_label]
    query = PROMPTS_SIB200["query"][prompt_language].format(
        labels=", ".join(SIB200_LABELS), text=line["text"]
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=SIB200_LABELS,
        gold_index=SIB200_LABELS.index(gold),
        instruction=PROMPTS_SIB200["instruction"][prompt_language],
    )


def prompt_sentiment(line, task_name=None):
    gold = line["sentiment_categories"][0]
    query = PROMPTS_SENTIMENT["query"][prompt_language].format(text=line["text"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=SENTIMENT_LABELS,
        gold_index=SENTIMENT_LABELS.index(gold) if gold in SENTIMENT_LABELS else 0,
        instruction=PROMPTS_SENTIMENT["instruction"][prompt_language],
    )


def prompt_urgency(line, task_name=None):
    full_text = f"{line['subject']}\n{line['body']}"
    eng_label = line["priority"]
    gold = URGENCY_LABEL_MAP[eng_label]
    query = PROMPTS_URGENCY["query"][prompt_language].format(text=full_text)
    return Doc(
        task_name=task_name,
        query=query,
        choices=URGENCY_LABELS,
        gold_index=URGENCY_LABELS.index(gold),
        instruction=PROMPTS_URGENCY["instruction"][prompt_language],
    )


def prompt_text_tagging(line, task_name=None):
    query = PROMPTS_TEXT_TAGGING["query"][prompt_language].format(text=line["text"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[],
        gold_index=0,
        instruction=PROMPTS_TEXT_TAGGING["instruction"][prompt_language],
        specific={"gold_topics": line["keywords"]},
    )


prompt_language = "en"

PROMPTS_INSTRUCTION = {
    "instruction": {
        "hy": "Կարդա պահանջը և տրամադրված կոնտեքստը և պատասխանիր հարցին։\n",
        "en": "Read the instruction and the provided context and answer the question.\n",
    },
    "query": {
        "hy": "Պահանջ\n{instruction}\n\nԿոնտեքստ\n{context}",
        "en": "Instruction: {instruction}\nContext: {context}",
    },
}

PROMPTS_QA_CONTEXT = {
    "instruction": {
        "hy": "Օգտվելով տեքստից` պատասխանիր հարցին։\n",
        "en": "Answer the question using the context.\n",
    },
    "query": {
        "hy": "Տեքստ\n{context}\n\nՀարց\n{question}",
        "en": "Context: {context}\nQuestion: {question}",
    },
}


def prompt_alpaca(line, task_name=None):
    context = line.get("input", "")
    query = PROMPTS_INSTRUCTION["query"][prompt_language].format(
        instruction=line["instruction"], context=context
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["output"]],
        gold_index=0,
        instruction=PROMPTS_INSTRUCTION["instruction"][prompt_language],
    )


def prompt_databricks(line, task_name=None):
    context = line.get("context", "")
    query = PROMPTS_INSTRUCTION["query"][prompt_language].format(
        instruction=line["instruction"], context=context
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["response"]],
        gold_index=0,
        instruction=PROMPTS_INSTRUCTION["instruction"][prompt_language],
    )


def prompt_ms_marco(line, task_name=None):
    raw = line["armenian"]

    question, context, answers = "", "", ""
    section = None
    buffer = []

    for l in raw.splitlines():
        if l.startswith("INPUT:"):
            if section == "ANSWERS":
                answers = "\n".join(buffer).strip()
            elif section == "CONTEXT":
                context = "\n".join(buffer).strip()
            elif section == "INPUT":
                question = "\n".join(buffer).strip()
            section, buffer = "INPUT", [l.replace("INPUT:", "").strip()]
        elif l.startswith("CONTEXT:"):
            if section == "INPUT":
                question = "\n".join(buffer).strip()
            elif section == "ANSWERS":
                answers = "\n".join(buffer).strip()
            section, buffer = "CONTEXT", [l.replace("CONTEXT:", "").strip()]
        elif l.startswith("ANSWERS:"):
            if section == "CONTEXT":
                context = "\n".join(buffer).strip()
            elif section == "INPUT":
                question = "\n".join(buffer).strip()
            section, buffer = "ANSWERS", [l.replace("ANSWERS:", "").strip()]
        else:
            buffer.append(l.strip())

    if section == "INPUT":
        question = "\n".join(buffer).strip()
    elif section == "CONTEXT":
        context = "\n".join(buffer).strip()
    elif section == "ANSWERS":
        answers = "\n".join(buffer).strip()

    query = PROMPTS_QA_CONTEXT["query"][prompt_language].format(
        context=context, question=question
    )
    return Doc(
        instruction=PROMPTS_QA_CONTEXT["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=[answers],
        gold_index=0,
        specific={"gold": [answers]},
    )


def prompt_squad(line, task_name=None):
    context = line["context"]
    question = line["question"]
    query = PROMPTS_QA_CONTEXT["query"][prompt_language].format(
        context=context, question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["answer"]],
        gold_index=0,
        instruction=PROMPTS_QA_CONTEXT["instruction"][prompt_language],
    )


from lighteval.metrics.metrics import Metrics

prompt_language = "en"
PROMPTS_CONTEXT_MCQA = {
    "instruction": {
        "hy": "Ընտրիր ճիշտ պատասխանը տրված տարբերակներից` օգտվելով կոնտեքստից։\n",
        "en": "Choose the correct answer from the given options using the context.\n",
    },
    "query": {
        "hy": ("Կոնտեքստ\n{context}\n\n" "Հարց\n{question}\n" "Ճիշտ տարբերակ:"),
        "en": ("Context: {context}\n" "Question: {question}\n" "Correct Option:"),
    },
}

PROMPTS_DREAM = {
    "instruction": {
        "hy": "Պատասխանեք հարցին՝ հիմնվելով երկխոսության վրա։\n",
        "en": "Answer the question based on the dialogue.\n",
    },
    "query": {
        "hy": ("Երկխոսություն\n {dialogue}\n\n" "Հարց\n{question}\n" "Ճիշտ տարբերակ:"),
        "en": ("Dialogue:\n{dialogue}\n" "Question: {question}\n" "Correct Option:"),
    },
}


def prompt_belebele(line, task_name=None):
    passage = line["flores_passage"]
    question = line["question"]
    choices = [
        line["mc_answer1"],
        line["mc_answer2"],
        line["mc_answer3"],
        line["mc_answer4"],
    ]
    gold_index = line["correct_answer_num"] - 1
    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(
        context=passage, question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=PROMPTS_CONTEXT_MCQA["instruction"][prompt_language],
    )


def prompt_scientific(line, task_name=None):
    context = line["context"]
    question = line["question"]
    choices = line["choices"]
    gold_index = line["gold_index"]

    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(
        context=context, question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=PROMPTS_CONTEXT_MCQA["instruction"][prompt_language],
    )


def prompt_syndarin(line, task_name=None):
    context = line["paragraph"]
    question = line["question"]
    choices = [
        line["answer_candidate_1"],
        line["answer_candidate_2"],
        line["answer_candidate_3"],
        line["answer_candidate_4"],
    ]
    gold_index = choices.index(line["correct_answer"])
    query = PROMPTS_CONTEXT_MCQA["query"][prompt_language].format(
        context=context, question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=PROMPTS_CONTEXT_MCQA["instruction"][prompt_language],
    )


def prompt_dream(line, task_name=None):
    dialogue = line["dialogue"]
    if isinstance(dialogue, list):
        dialogue = "\n".join(dialogue)

    question = line["question"]
    choices = line["choices"]
    gold_index = line["label"]

    query = PROMPTS_DREAM["query"][prompt_language].format(
        dialogue=dialogue, question=question
    )
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=gold_index,
        instruction=PROMPTS_DREAM["instruction"][prompt_language],
    )


from lighteval.metrics.metrics import Metrics

prompt_language = "en"


PROMPTS_MCQA = {
    "instruction": {
        "hy": "Պատասխանիր հարցին՝ ընտրելով ճիշտ տարբերակը։\n",
        "en": "Answer the question by choosing the correct option.\n",
    },
    "query": {
        "hy": "{question}\n" "Ճիշտ տարբերակ:",
        "en": "{question}\n" "Correct Option:",
    },
}


def prompt_hartak_mcqa(line, task_name=None):
    choices = [line["answer"]] + line["distractors"]
    query = PROMPTS_MCQA["query"][prompt_language].format(question=line["question"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in choices],
        gold_index=0,
        instruction=PROMPTS_MCQA["instruction"][prompt_language],
    )


def prompt_include(line, task_name=None):
    choices = [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]
    query = PROMPTS_MCQA["query"][prompt_language].format(question=line["question"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[f" {c}" for c in choices],
        gold_index=line["answer"],
        instruction=PROMPTS_MCQA["instruction"][prompt_language],
    )


from .ner_span_metric import ner_span_metric
from .pos_metric import pos_metric

prompt_language = "en"
UPOS_TAGS = [
    "Գոյական",
    "Ածական",
    "Բայ",
    "Մակբայ",
    "Թիվ",
    "Դերանուն",
    "Կապ / Կապակցիչ",
    "Մասնիկ",
    "Ձայնարկություն",
]

BIO_TAGS = ["PER", "ORG", "LOC"]

PROMPTS_NER = {
    "instruction": {
        "hy": (
            "Տեքստում գտիր անվանական սուբյեկտները (named entities) և նշիր դրանց "
            "տեսակը օգտվելով միայն առկա թեգերի ցանկից։ Պատասխանը վերադարձրու JSON array ֆորմատով, "
            "որտեղ յուրաքանչյուր օբյեկտ ունի 'tag' և 'entity' բանալիներ: Վերադարձրու միայն մեկ JSON array, "
            "առանց կրկնությունների: Եթե տեքստում չկան թեգեր, ապա վերադարձրու '[]':\n"
        ),
        "en": (
            "Find the named entities in the text and specify their type using only the provided tags list. "
            "Return the answer as a single JSON array where each object has 'tag' and 'entity' keys. "
            "Return only one JSON array without repetition. "
            "If there are no named entities in the text, return '[]':\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ\n{text}\n\n" "Հնարավոր թեգեր՝ {tags}"),
        "en": ("Text: {text}\n" "Possible tags: {tags}\n" ""),
    },
}

PROMPTS_UPOS = {
    "instruction": {
        "hy": (
            "Տրված բառի համար որոշիր ճիշտ խոսքի մասը՝ օգտվելով միայն առկա խոսքի մասերի ցանկից։ "
            'Պատասխանը գրիր այս ձևաչափով՝ "Այս բառի խոսքի մասն է <խոսքի մաս>"\n'
        ),
        "en": (
            "For the given word, determine the correct part of speech using the provided list.\n"
            'Write the answer in this format: "The part of speech of this word is <part of speech>".\n'
        ),
    },
    "query": {
        "hy": ("Տրված բառը՝ '{form}'\n" "Խոսքի մասեր՝ {tags}\n"),
        "en": ("Given word: '{form}'\n" "Parts of speech: {tags}\n"),
    },
}


def prompt_finer(line, task_name=None):
    gold = [(t, ty) for t, ty in line["gold_entities"]]
    tag_pool = sorted(set(ty for _, ty in gold))
    query = PROMPTS_NER["query"][prompt_language].format(
        text=line["text"], tags=", ".join(tag_pool)
    )
    return Doc(
        instruction=PROMPTS_NER["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_pioner(line, task_name=None):
    tokens, tags = line["tokens"], line["ner_tags"]
    gold, current, current_type = [], [], None

    for tok, tag in zip(tokens, tags):
        if tag != "O":
            if current_type == tag:
                current.append(tok)
            else:
                if current:
                    gold.append((" ".join(current), current_type))
                current, current_type = [tok], tag
        else:
            if current:
                gold.append((" ".join(current), current_type))
                current, current_type = [], None
    if current:
        gold.append((" ".join(current), current_type))

    query = PROMPTS_NER["query"][prompt_language].format(
        text=" ".join(tokens),
        tags=", ".join(BIO_TAGS),
    )
    return Doc(
        instruction=PROMPTS_NER["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_ud_armtdp(line, task_name=None):
    word = line["form"]
    gold = [(word, str(line["upos_hy"]))] if line.get("upos_hy") else []
    query = PROMPTS_UPOS["query"][prompt_language].format(
        form=word, tags=", ".join(UPOS_TAGS)
    )
    return Doc(
        instruction=PROMPTS_UPOS["instruction"][prompt_language],
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        specific={"gold_entities": gold},
    )


def prompt_alpaca(line, task_name=None):
    return Doc(
        task_name=task_name,
        query=line["instruction"],
        choices=[line["output"]],
        gold_index=0,
        instruction="",
    )


def prompt_databricks(line, task_name=None):
    return Doc(
        task_name=task_name,
        query=line["instruction"],
        choices=[line["response"]],
        gold_index=0,
        instruction="",
    )


def prompt_qa(line, task_name=None):
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[line["answer"]],
        gold_index=0,
        instruction="",
    )


def prompt_factual_memorisation(line, task_name=None):
    """Prompt function for factual-memorisation subset.
    Dataset structure: {'collection': 'eng'|'factual'|'memorisation', 'prompt': '...', 'completion': '...'}
    """
    query = f"Question: {line['prompt']}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["completion"]],
        gold_index=0,
        instruction="",
        specific={"collection": line.get("collection", "unknown")},
    )


from lighteval.metrics.metrics import Metrics
from .summ_para_translation_metric import BertScoreArm, ParaphraseBestMatch

prompt_language = "en"  # or 'hy'

PROMPTS_EMAIL_SUM = {
    "instruction": {
        "hy": "Ամփոփիր էլեկտրոնային նամակը {num_sents} նախադասությամբ։\n",
        "en": "Generate a summary for the email in {num_sents} sentences.\n",
    },
    "query": {"hy": "Էլեկտրոնային նամակ\n{email}", "en": "Email: {email}"},
}

PROMPTS_CONV_SUM = {
    "instruction": {
        "hy": "Ամփոփիր երկխոսությունը {num_sents} նախադասությամբ։\n",
        "en": "Generate a summary for the dialogue in {num_sents} sentences.\n",
    },
    "query": {"hy": "Երկխոսություն\n{dialogue}", "en": "Dialogue:\n{dialogue}"},
}

PROMPTS_PARAPHRASE = {
    "instruction": {
        "hy": "Վերաշարադրիր նախադասությունը և վերադարձու միայն փոփոխված տեքստը։\n",
        "en": "Paraphrase the sentence and return only the paraphrased text.\n",
    },
    "query": {"hy": "{text}", "en": "{text}"},
}

PROMPTS_TRANSLATION = {
    "instruction": {
        "hy": "Թարգմանիր տրված անգլերեն տեքստը հայերեն և վերադարձու միայն թարգմանված տարբերակը։\n",
        "en": "Translate the given English text into Armenian and return the translated text only.\n",
    },
    "query": {
        "hy": "Անգլերեն: {eng}\n\n" "Հայերեն:",
        "en": "English: {eng}\n\n" "Armenian:",
    },
}


def count_sentences(text: str) -> int:
    if not text:
        return 0
    normalized = text.replace(":", "։")
    parts = [p.strip() for p in normalized.split("։") if p.strip()]
    return len(parts)


def prompt_email_sum(line, task_name=None):
    num_sents = count_sentences(line["summary"])
    query = PROMPTS_EMAIL_SUM["query"][prompt_language].format(email=line["email"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["summary"]],
        gold_index=0,
        instruction=PROMPTS_EMAIL_SUM["instruction"][prompt_language].format(
            num_sents=num_sents
        ),
    )


def prompt_conv_sum(line, task_name=None):
    num_sents = count_sentences(line["summary"])
    query = PROMPTS_CONV_SUM["query"][prompt_language].format(dialogue=line["dialogue"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["summary"]],
        gold_index=0,
        instruction=PROMPTS_CONV_SUM["instruction"][prompt_language].format(
            num_sents=num_sents
        ),
    )


def prompt_paraphrase(line, task_name=None):
    query = PROMPTS_PARAPHRASE["query"][prompt_language].format(text=line["text"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=line["paraphrases"],
        gold_index=0,
        instruction=PROMPTS_PARAPHRASE["instruction"][prompt_language],
    )


def prompt_translation(line, task_name=None):
    query = PROMPTS_TRANSLATION["query"][prompt_language].format(eng=line["eng"])
    return Doc(
        task_name=task_name,
        query=query,
        choices=[line["hy"]],
        gold_index=0,
        instruction=PROMPTS_TRANSLATION["instruction"][prompt_language],
    )


import numpy as np
import json
import re
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
    SampleLevelMetricGrouping,
    SamplingMethod,
    SampleLevelComputation,
)

prompt_language = "en"
PUNCTUATION_CHARS = set([",", "՝", ":", "։", "`"])

PROMPTS_SPACE_FIX = {
    "instruction": {
        "hy": (
            "Ավելացրու տեքստում պակաս բացատները և վերադարձրու ուղղված տարբերակը "
            "առանց որևէ մեկնաբանության և բացատրության։\n"
        ),
        "en": (
            "Add the missing spaces in the text and return only the corrected version without any comments or explanations.\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ: {corrupted}\n" "Պատասխան:"),
        "en": ("Text: {corrupted}\n" "Answer:"),
    },
}

PROMPTS_PUNCTUATION = {
    "instruction": {
        "hy": (
            "Ուղղիր (ավելացրու) տեքստում բաց թողնված կետադրական նշանները և վերադարձրու "
            "միայն ուղղված տարբերակը առանց որևէ մեկնաբանության և բացատրության։\n"
        ),
        "en": (
            "Correct (add) the missing punctuation marks in the text and return only the corrected version without any comments or explanations.\n"
        ),
    },
    "query": {
        "hy": ("Տեքստ: {corrupted}\n" "Պատասխան:"),
        "en": ("Text: {corrupted}\n" "Answer:"),
    },
}


def mean_corpus_level(items):
    return float(np.mean(items)) if items else 0.0


def extract_fixed_text(model_response):
    raw_text = (
        model_response.text_post_processed[0]
        if hasattr(model_response, "text_post_processed")
        and model_response.text_post_processed
        else (
            model_response.text[0]
            if hasattr(model_response, "text") and model_response.text
            else ""
        )
    ).strip()

    match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "ուղղված տարբերակ" in parsed:
                    return parsed["ուղղված տարբերակ"].strip()
                elif "corrected version" in parsed:
                    return parsed["corrected version"].strip()
        except Exception:
            pass
    return raw_text


def prompt_space_fix(line, task_name=None):
    gold = line["gold"]
    corrupted = line["corrupted_spaces"]
    query = PROMPTS_SPACE_FIX["query"][prompt_language].format(corrupted=corrupted)
    return Doc(
        task_name=task_name,
        query=query,
        choices=None,
        gold_index=None,
        instruction=PROMPTS_SPACE_FIX["instruction"][prompt_language],
        specific={"gold": gold},
    )


def prompt_punctuation(line, task_name=None):
    gold = line["gold"]
    corrupted = line["corrupted_punctuation"]
    query = PROMPTS_PUNCTUATION["query"][prompt_language].format(corrupted=corrupted)
    return Doc(
        task_name=task_name,
        query=query,
        choices=[gold],
        gold_index=0,
        instruction=PROMPTS_PUNCTUATION["instruction"][prompt_language],
    )


class SpaceAccuracyComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs) -> float:
        gold = doc.specific["gold"].strip()
        pred = extract_fixed_text(model_response)
        gold_words, pred_words = gold.split(), pred.split()
        matched = 0
        for g in gold_words:
            if not pred_words:
                break
            if g in pred_words:
                pred_words.remove(g)
                matched += 1
                continue
            found_idx = next(
                (i for i, p in enumerate(pred_words) if p.endswith(g)), None
            )
            if found_idx is not None:
                matched += 1
                pred_words.pop(found_idx)
        return matched / len(gold_words) if gold_words else 0.0


class PunctuationAccuracyComputation(SampleLevelComputation):
    def compute(self, model_response, doc, **kwargs) -> float:
        gold = doc.choices[0].strip()
        pred = extract_fixed_text(model_response).strip()

        if gold == pred:
            return 1.0

        def remove_punct(text: str):
            return "".join(ch for ch in text if ch not in PUNCTUATION_CHARS)

        gold_nopunct = remove_punct(gold)
        pred_nopunct = remove_punct(pred)

        if gold_nopunct != pred_nopunct:
            return 0.0

        gold_words, pred_words = gold.split(), pred.split()

        if len(gold_words) != len(pred_words):
            return 0.0

        total_punctuated, correct = 0, 0
        for gw, pw in zip(gold_words, pred_words):
            if any(ch in PUNCTUATION_CHARS for ch in gw):
                total_punctuated += 1
                if gw == pw:
                    correct += 1

        return correct / total_punctuated if total_punctuated > 0 else 1.0


space_accuracy_metric = SampleLevelMetric(
    metric_name="space_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=SpaceAccuracyComputation(),
    corpus_level_fn=mean_corpus_level,
)

punctuation_accuracy_metric = SampleLevelMetric(
    metric_name="punctuation_accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=PunctuationAccuracyComputation(),
    corpus_level_fn=mean_corpus_level,
)


# Custom metric for factual-memorisation with collection-based scoring
_collection_scores_storage = (
    []
)  # Store (score, collection) pairs for corpus-level aggregation
_collection_scores_cache: dict[str, list[float]] = (
    {}
)  # Cache for corpus-level aggregation


def _token_set(text: str) -> set[str]:
    """Extract tokens from text."""
    return {token for token in text.lower().split() if token}


class FactualMemorisationComputation(SampleLevelComputation):
    """Compute word overlap for eng/factual, bigram overlap for memorisation."""

    def compute(self, model_response, doc, **kwargs) -> dict:
        collection = doc.specific.get("collection", "unknown")
        gold = doc.choices[0].strip() if doc.choices else ""

        # Get prediction from model response
        if (
            hasattr(model_response, "text_post_processed")
            and model_response.text_post_processed
        ):
            pred = model_response.text_post_processed[0].strip()
        elif hasattr(model_response, "text") and model_response.text:
            pred = model_response.text[0].strip()
        else:
            pred = ""

        # Initialize all metrics to 0.0
        eng_score = 0.0
        factual_score = 0.0
        memorisation_score = 0.0

        if collection == "memorisation":
            # Use bigram overlap for memorisation (from evaluate_memorisation_bigram_overlap)
            reference_tokens = gold.lower().split()
            predicted_tokens = pred.lower().split()

            reference_bigrams = {
                (reference_tokens[i], reference_tokens[i + 1])
                for i in range(len(reference_tokens) - 1)
            }
            predicted_bigrams = {
                (predicted_tokens[i], predicted_tokens[i + 1])
                for i in range(len(predicted_tokens) - 1)
            }

            if not reference_bigrams:
                memorisation_score = 1.0
            else:
                overlap = sum(
                    bigram in predicted_bigrams for bigram in reference_bigrams
                )
                memorisation_score = overlap / len(reference_bigrams)
        else:
            # Use word overlap for eng and factual (from evaluate_word_overlap)
            reference_tokens = _token_set(gold)
            predicted_tokens = _token_set(pred)

            if not reference_tokens:
                score = 1.0
            else:
                score = sum(
                    token in predicted_tokens for token in reference_tokens
                ) / len(reference_tokens)

            # Assign score based on collection type
            if collection == "eng":
                eng_score = score
            elif collection == "factual":
                factual_score = score

        # Store score with collection for corpus-level aggregation
        # Get the score for the current collection type
        if collection == "eng":
            score = eng_score
        elif collection == "factual":
            score = factual_score
        elif collection == "memorisation":
            score = memorisation_score
        else:
            score = 0.0
        _collection_scores_storage.append((score, collection))
        # Clear cache when new score is added (will be rebuilt on corpus-level call)
        _collection_scores_cache.clear()

        # Return dictionary with all three metrics
        return {
            "eng_overlap": eng_score,
            "factual_overlap": factual_score,
            "memorisation_overlap": memorisation_score,
        }


def get_collection_scores_by_type(collection_type: str):
    """Helper to get scores for a specific collection type."""
    if not _collection_scores_cache:
        # Build cache from storage
        for score, collection in _collection_scores_storage:
            _collection_scores_cache.setdefault(collection, []).append(score)
    return _collection_scores_cache.get(collection_type, [])


def eng_overlap_corpus_level(sample_scores: list[float]) -> float:
    """Corpus-level aggregation for eng collection."""
    scores = get_collection_scores_by_type("eng")
    return float(np.mean(scores)) if scores else 0.0


def factual_overlap_corpus_level(sample_scores: list[float]) -> float:
    """Corpus-level aggregation for factual collection."""
    scores = get_collection_scores_by_type("factual")
    return float(np.mean(scores)) if scores else 0.0


def memorisation_overlap_corpus_level(sample_scores: list[float]) -> float:
    """Corpus-level aggregation for memorisation collection."""
    scores = get_collection_scores_by_type("memorisation")
    return float(np.mean(scores)) if scores else 0.0


factual_memorisation_metric = SampleLevelMetricGrouping(
    metric_name=["eng_overlap", "factual_overlap", "memorisation_overlap"],
    higher_is_better={
        "eng_overlap": True,
        "factual_overlap": True,
        "memorisation_overlap": True,
    },
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=FactualMemorisationComputation(),
    corpus_level_fn={
        "eng_overlap": eng_overlap_corpus_level,
        "factual_overlap": factual_overlap_corpus_level,
        "memorisation_overlap": memorisation_overlap_corpus_level,
    },
)


class ArmenianEvalTask(LightevalTaskConfig):
    def __init__(
        self,
        short_name,
        hf_subset,
        prompt_function,
        metrics,
        generation=False,
        few_shots_split=None,
        few_shots_select=None,
        evaluation_splits=None,
        hf_avail_splits=None,
    ):
        # Default to ["train"], but allow override for subsets that only have ["test"]
        default_eval_splits = (
            evaluation_splits if evaluation_splits is not None else ["train"]
        )
        default_hf_splits = (
            hf_avail_splits if hf_avail_splits is not None else ["train"]
        )

        super().__init__(
            name=f"armenian_evals:{short_name}",
            prompt_function=prompt_function,
            suite=["community"],
            hf_repo="Metric-AI/HY-benchmark-ds-clean",
            hf_subset=hf_subset,
            metrics=metrics,
            hf_avail_splits=default_hf_splits,
            evaluation_splits=default_eval_splits,
            few_shots_split=few_shots_split,
            few_shots_select=few_shots_select,
            generation_size=2048 if generation else -1,  # ✅ unified rule
            stop_sequence=None,
        )


TASKS_TABLE = [
    # Classification
    ArmenianEvalTask(
        "topic-14class", "topic-14class", prompt_sib200, [Metrics.loglikelihood_acc]
    ),
    # ArmenianEvalTask("sentiment", "sentiment", prompt_sentiment, [Metrics.loglikelihood_acc]),
    # ArmenianEvalTask("urgency", "urgency", prompt_urgency, [Metrics.loglikelihood_acc]),
    # ArmenianEvalTask("text_tagging", "text_tagging", prompt_text_tagging, [text_tagging_metric], generation=True),
    # Text editing
    ArmenianEvalTask(
        "space_fix",
        "space_fix",
        prompt_space_fix,
        [space_accuracy_metric],
        generation=True,
    ),
    ArmenianEvalTask(
        "punctuation",
        "punctuation",
        prompt_punctuation,
        [punctuation_accuracy_metric],
        generation=True,
    ),
    # NER / POS
    # ArmenianEvalTask("finer", "finer", prompt_finer, [ner_span_metric]),
    ArmenianEvalTask(
        "pioner", "pioner", prompt_pioner, [ner_span_metric], generation=True
    ),
    ArmenianEvalTask("pos", "pos", prompt_ud_armtdp, [pos_metric], generation=True),
    # Simple QA
    ArmenianEvalTask("arak", "simpleqa", prompt_qa, [Metrics.bleu], generation=True),
    ArmenianEvalTask(
        "factual-memorisation",
        "factual-memorisation",
        prompt_factual_memorisation,
        [factual_memorisation_metric],
        generation=True,
        evaluation_splits=["test"],
        hf_avail_splits=["test"],
    ),
    # ArmenianEvalTask("alpaca_simple", "alpaca-no-context-instr-following", prompt_alpaca, [Metrics.bleu], generation=True),
    # ArmenianEvalTask("databricks_simple", "databricks-no-context-instr-following", prompt_databricks, [Metrics.bleu], generation=True),
    # InContext QA
    # ArmenianEvalTask("alpaca", "alpaca-instr-following", prompt_alpaca, [Metrics.bleu], generation=True),
    # ArmenianEvalTask("databricks", "databricks-instr-following", prompt_databricks, [Metrics.bleu], generation=True),
    ArmenianEvalTask(
        "ms_marco",
        "ms-marco-in-context-qa",
        prompt_ms_marco,
        [Metrics.bleu],
        generation=True,
    ),
    ArmenianEvalTask(
        "squad", "squad-in-context-qa", prompt_squad, [Metrics.bleu], generation=True
    ),
    # InContext MCQA
    ArmenianEvalTask(
        "belebele",
        "belebele-in-context-mcqa",
        prompt_belebele,
        [Metrics.loglikelihood_acc],
    ),
    ArmenianEvalTask(
        "scientific",
        "scientific-in-context-mcqa",
        prompt_scientific,
        [Metrics.loglikelihood_acc],
    ),
    ArmenianEvalTask(
        "syndarin",
        "syndarin-in-context-mcqa",
        prompt_syndarin,
        [Metrics.loglikelihood_acc],
    ),
    ArmenianEvalTask(
        "dream", "conversation-in-context-qa", prompt_dream, [Metrics.loglikelihood_acc]
    ),
    # MCQA
    ArmenianEvalTask(
        "include", "include-mcqa", prompt_include, [Metrics.loglikelihood_acc]
    ),
    # ArmenianEvalTask("hartak", "public-services-mcqa", prompt_hartak_mcqa, [Metrics.loglikelihood_acc]),
    # Summ/Paraphrase/Translation
    ArmenianEvalTask(
        "email",
        "email-sum",
        prompt_email_sum,
        [Metrics.bleu, Metrics.bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "conversation",
        "conversational-sum",
        prompt_conv_sum,
        [Metrics.bleu, Metrics.bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "paraphrase",
        "paraphrase",
        prompt_paraphrase,
        [Metrics.bleu, Metrics.bert_score_arm],
        generation=True,
    ),
    ArmenianEvalTask(
        "short_sentences_translation",
        "translation_short_sentences",
        prompt_translation,
        [Metrics.bleu, Metrics.bert_score_arm],
        generation=True,
    ),
]
