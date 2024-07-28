import streamlit as st
from pypdf import PdfReader
from transformers import AutoTokenizer, TFBartForConditionalGeneration


@st.cache_resource
def load_summarizer() :
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model


def summarize_text(text, num_sentences) :
    tokenizer, model = load_summarizer()
    inputs = tokenizer.encode("summarize: " + text, return_tensors="tf", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=num_sentences * 30, min_length=num_sentences * 20,
                                 length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def summarizer(pdf, num_sentences) :
    if pdf is not None :
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages :
            text += page.extract_text() or ""

        summary = summarize_text(text, num_sentences)
        return summary
    return "No PDF content found to summarize."


def main() :
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄")

    st.title("📚 PDF Summarizing App")
    st.markdown("Summarize your PDF files using the BART model from Hugging Face.")

    st.sidebar.header("About")
    st.sidebar.info("This app uses the BART model from Hugging Face to generate summaries of uploaded PDF documents.")

    st.divider()

    pdf = st.file_uploader("Upload your PDF Document", type='pdf')

    col1, col2 = st.columns(2)
    with col1 :
        num_sentences = st.slider("Approximate number of sentences in summary", min_value=1, max_value=10, value=3)
    with col2 :
        submit = st.button("Generate Summary", type="primary")

    if submit :
        if pdf is not None :
            with st.spinner('Generating summary... This may take a while for large documents.') :
                summary = summarizer(pdf, num_sentences)
                st.success("Summary generated successfully!")
                st.subheader("Summary:")
                st.markdown(f">{summary}")

                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )
        else :
            st.error("Please upload a PDF file.")

    st.divider()
    st.markdown("Made with ❤️ by Your Name")


if __name__ == "__main__" :
    main()
