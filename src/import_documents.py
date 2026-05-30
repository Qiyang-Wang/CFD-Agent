import argparse
from pathlib import Path
from pdf_chunk_ask_question import DocumentProcessor
from rag_database import RAGDatabase

def import_document(db, processor, file_path):
    if file_path.suffix.lower() == '.pdf':
        chunks = processor.process_pdf(str(file_path))
        db.add_document(chunks, "pdf", file_path)
    elif file_path.suffix.lower() == '.md':
        chunks = processor.process_markdown(str(file_path))
        db.add_document(chunks, "markdown", file_path)
    print(f"Imported {file_path} with {len(chunks)} chunks")

def main():
    parser = argparse.ArgumentParser(description="Import documents to RAG database")
    parser.add_argument("--dir", type=str, help="Directory containing documents")
    args = parser.parse_args()

    # 修改模型路径为本地已下载的模型
    processor = DocumentProcessor(model_name='BAAI/bge-m3')  # 使用test_env/download_model.py下载的模型
    db = RAGDatabase()
    
    for file_path in Path(args.dir).rglob("*.[pdf|md]"):
        import_document(db, processor, file_path)
    
    db.save()
    print(f"Database saved with {db.index.ntotal} embeddings")

if __name__ == "__main__":
    main()
