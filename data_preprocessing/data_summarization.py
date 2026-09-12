from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Text and Table Prompt
# ============================================================

prompt_text_table = """
You are an assistant tasked with summarizing tables and text.

Give a concise summary of the table or text.

Respond only with the summary, no additional comment.

Do not start your message by saying "Here is a summary"
or anything like that.

Just give the summary as it is.

Table or text chunk: {element}
"""

prompt_to_summarize_text_and_table = (
    ChatPromptTemplate.from_template(prompt_text_table)
)


# ============================================================
# Image Prompt
# ============================================================

prompt_image = """
Describe the image in detail.

For context, the image is part of a research paper
explaining the Transformers architecture.

Be specific about:
- diagrams
- graphs
- bar plots
- axes
- labels
- relationships between components
- important visual information

Focus on information that would be useful for
retrieval-augmented generation.

Respond only with the image description.
"""


# ============================================================
# Summarization
# ============================================================

def summarize_data(texts, tables, images):

    # --------------------------------------------------------
    # Gemini Model
    # --------------------------------------------------------

    model = ChatGoogleGenerativeAI(
        temperature=0.5,
        model="gemini-2.5-flash"
    )


    # ========================================================
    # TEXT + TABLE SUMMARIZATION
    # ========================================================

    text_table_summarize_chain = (
        {"element": lambda x: x}
        | prompt_to_summarize_text_and_table
        | model
        | StrOutputParser()
    )


    # --------------------------------------------------------
    # Summarize Text
    # --------------------------------------------------------

    print("Summarizing text...")

    text_summaries = text_table_summarize_chain.batch(
        texts,
        config={"max_concurrency": 3}
    )


    # --------------------------------------------------------
    # Summarize Tables
    # --------------------------------------------------------

    print("Summarizing tables...")

    tables_html = [
        table.metadata.text_as_html
        for table in tables
    ]

    table_summaries = text_table_summarize_chain.batch(
        tables_html,
        config={"max_concurrency": 3}
    )


    # ========================================================
    # IMAGE SUMMARIZATION
    # ========================================================

    print("Summarizing images...")


    # --------------------------------------------------------
    # Create multimodal prompt
    # --------------------------------------------------------

    image_summarize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": prompt_image
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            )
        ]
    )


    # --------------------------------------------------------
    # Image summarization chain
    # --------------------------------------------------------

    image_summarize_chain = (
        image_summarize_prompt
        | model
        | StrOutputParser()
    )


    # --------------------------------------------------------
    # Summarize Images
    # --------------------------------------------------------

    image_summaries = image_summarize_chain.batch(
        images,
        config={"max_concurrency": 3}
    )


    # ========================================================
    # Return Results
    # ========================================================

    return (
        text_summaries,
        table_summaries,
        image_summaries
    )