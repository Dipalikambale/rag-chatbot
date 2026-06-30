from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # Increased from 400 to capture entire paragraphs
        chunk_overlap=150,     # Increased from 50 to make sure concepts aren't cut in half
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)