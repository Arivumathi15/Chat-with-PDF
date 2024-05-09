**Title:** Enhancing Document Interaction: Chat with PDF using Gemini

**Description:**

"Chat with PDF using Gemini" is a groundbreaking project that redefines document interaction by integrating natural language processing (NLP) and artificial intelligence (AI). This innovative tool enables users to engage with PDF documents in a conversational manner, streamlining the process of extracting insights and locating specific information within lengthy documents.

**Key Features:**

**PDF Text Extraction:** Leveraging the PyPDF2 library, the project extracts text from uploaded PDF files, forming the basis for further analysis and interaction.
**Efficient Text Chunking:** Utilizing a Recursive Character Text Splitter, the tool divides extracted text into smaller, manageable chunks, enhancing processing speed and content analysis efficiency.
**Semantic Embedding:** The project harnesses Google Generative AI Embeddings to embed text chunks, capturing their semantic meaning. These embeddings facilitate similarity search and question-answering tasks.
**Vector Store Creation:** Embedded text chunks are stored in a vector store using FAISS, enabling rapid retrieval of relevant document segments based on user queries.
**Conversational AI:** Users can pose questions about PDF content using natural language. The integration of a Chat Google Generative AI model, trained on conversational data, generates responses based on user queries, enhancing the interactive experience.
**Keyword Highlighting:** A new feature has been added to highlight keywords or phrases within the PDF document related to the user's question. This functionality helps users quickly identify relevant information, improving the overall user experience.
