EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "instructor": "hkunlp/instructor-base",
}


def strategy_basic(text):
    return text.replace("\n", " ")


def strategy_normalized(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return " ".join(lines)


def strategy_minimal(text):
    return " ".join(text.split())


def strategy_preserve_structure(text):
    paragraphs = text.split("\n\n")
    cleaned = [" ".join(p.split()) for p in paragraphs if p.strip()]
    return "\n\n".join(cleaned)


def strategy_no_cleaning(text):
    return text


CLEANING_STRATEGIES = {
    "basic": strategy_basic,
    "normalized": strategy_normalized,
    "minimal": strategy_minimal,
    "preserve_structure": strategy_preserve_structure,
    "no_cleaning": strategy_no_cleaning,
}

CHUNK_CONFIGS = {
    "default": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "separators": ["\n\n", "\n", ".", " ", ""],
    },
    "small": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separators": ["\n\n", "\n", ".", " ", ""],
    },
    "large": {
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "separators": ["\n\n", "\n", ".", " ", ""],
    },
    "semantic": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "separators": [
            "\n\n\n",
            "\n\n",
            ".\n",
            ".",
            " ",
            "",
        ],
    },
}

TEST_QUERIES = [
    "What are the requirements for doctoral studies?",
    "How do I enroll in courses?",
    "What is the thesis submission process?",
    "What are the language requirements?",
    "Physical education requirements",
    "How to apply for joint PhD programs?",
    "What is the comprehensive examination?",
    "Internship requirements for students",
    "How to register for a semester?",
    "What is the deadline for course registration?",
    "How can I request a passive semester?",
    "What are the credit requirements for graduation?",
    "How to transfer credits from another university?",
    "How does the grading system work?",
    "What happens if I fail an exam?",
    "How many times can I retake an exam?",
    "What is the procedure for exam appeals?",
    "What scholarships are available for students?",
    "How to apply for tuition fee waiver?",
    "What are the payment deadlines for tuition?",
    "How to get a student ID card?",
    "What is Neptun and how to use it?",
    "How to contact the student office?",
    "What are the requirements for the Computer Science program?",
    "How long is the Master's program?",
    "What is the difference between BSc and MSc?",
    "How to change my personal data?",
    "What documents do I need for enrollment?",
    "How to get an official transcript?",
    "What is the final examination procedure?",
    "How to apply for diploma issuance?",
    "What are the requirements for obtaining a degree?",
]

DATA_PATH = "data/raw"
EXPERIMENT_BASE_PATH = "data/experiments"
