import gradio as gr
import requests

API_URL = "http://localhost:8000"

def upload_files(files):
    files_payload = [("files", (f.name, open(f.name, "rb"))) for f in files]
    response = requests.post(f"{API_URL}/upload", files=files_payload)
    return response.json()

def chat(query):
    response = requests.post(f"{API_URL}/chat", json={"query": query})
    return response.json()["response"]

with gr.Blocks(css="""
.answer-box {
    background-color: #eef4ff;
    color: #1a1a1a;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ccd9ff;
    white-space: pre-wrap;
    font-size: 16px;
}
.question-box textarea {
    background-color: #f7f7f7 !important;
    color: #1a1a1a !important;
    border-radius: 8px;
    border: 1px solid #ccc;
}
.header-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 5px;
}
.header-sub {
    text-align: center;
    color: #555;
    margin-bottom: 25px;
}
""") as demo:

    gr.Markdown("""
    <h1 style='text-align:center;'>📘 RAG Knowledge Assistant</h1>
    <p style='text-align:center;color:gray;'>Upload documents → Ask → Get contextual AI answers</p>
    """)

    with gr.Tab("📤 Upload Documents"):
        gr.Markdown("### Add new knowledge to your assistant")
        files = gr.File(file_count="multiple", label="Upload files (PDF / DOCX / TXT)")
        upload_btn = gr.Button("Upload & Process", variant="primary")
        upload_output = gr.JSON(label="Upload result")

        upload_btn.click(upload_files, inputs=files, outputs=upload_output)

    with gr.Tab("💬 Ask a Question"):
        gr.Markdown("### Chat with your AI assistant")

        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="Ask something about your documents...",
                elem_classes=["question-box"]
            )
            ask_btn = gr.Button("Ask", variant="primary")

        answer = gr.HTML("<div class='answer-box'>...</div>")

        def render_answer(q):
            response = chat(q)
            return f"<div class='answer-box'>{response}</div>"

        ask_btn.click(render_answer, inputs=question, outputs=answer)
demo.launch()
