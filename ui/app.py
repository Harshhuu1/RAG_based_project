from streamlit import  st
import httpx
import time
from datetime import datetime

#----App Config_____

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expand",
)

#____ Constants_______

API_URL="http://localhost:8000"


#____Session state____
if "messages" not in st.session_state:
    st.session_state.messages=[]

if "total_queries" not in st.session_state:
    st.session_state.total_queries=0

if "avg_latency" not in st.session_state:
    st.session_state.avg_latency=0.0

if "latencies" not in st.session_state:
    st.session_state.latencies=[]


#---- sidebar

with st.sidebar:
    st.title("DocMind")
    st.markdown("----")

    #File Uploader
    st.subheader(" Upload Documents")
    uplaoded_file= st.file_uploader(
        "Upload a document",
        type=["Pdf","txt", "docx","md"],
    )

    if uploaded_file:
        if st.button(" Ingest Document"):
            with st.spinner("Ingesting document..."):
                files={"file":(uploaded_file.name,uplaoded_file,uplaoded_file.type)}    
                response=httpx.post(f"{API_URL}/upload",files=files)

                if response.status_code==200:
                    data=response.json()
                    st.success(f"Ingested Successfully!")
                    st.metric("Chunks creater ", data["total_chunks"])
                else:
                    st.error(f" Ingestion failed: {response.txt}")
    st.markdown("----")

    #metrics

    st.subheader(" Session metrics")
    st.metric("total queries", st.session_state.total_queries)
    st.metric("Avg Latency", f"{st.session_state.avg_latency:.0f}ms")

    st.markdown("-----")

    #links
    st.subhead("Links")
    st.markdown("- [API Docs](http://localhost:8000/docs)")
    st.markdown("- [MLflow](http://localhost:5000)")
    st.markdown("- [Grafana](http://localhost:3000)")

#___Main chat interface___
st.title(" DocMind- Agentic RAG System")
st.markdown("Ask anhything about documents or any topic!")
st.markdown("---")

#Display chat history

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f" {message['latency']:.0f}ms")
#chat input

if prompt:= st.chat_input("Ask a question..."):

    #add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    #display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #get agent response
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            start=time.time()
        
        try:
            response= httpx.post(
                f"{API_URL}/query",
                json={"question":prompt},
                timeout=60.0,
            )

            if response.status_code==200:
                data=response.json()
                answer=data["answer"]
                latency =data["latency_ms"]

                #Display answer

                st.markdown(answer)
                st.caption(f" {latency:.0f} ms")

                #update session metrics

                st.session_state.total_queries +=1
                st.session_state.latencies.append(latency)
                st.session_state.avg_latency=sum(
                    st.session_state.latencies
                ) /len(st.session_state.latencies)

            #add to history

                st.session_state.messages.append({
                    "role":"assistant",
                    "latency":answer,
                    "latency": latency,
                })

            else:
                st.error(f" Error:{response.text}")
        except Exception as e:
            st.error(f" Failed to connect to API: {e}")
            
# What this means:

# st.chat_message("user") → renders a chat bubble with user avatar
# st.chat_input → the text input box at the bottom of the page
# := → walrus operator, assigns and checks in one line
# timeout=60.0 → wait up to 60 seconds for agent response since it may call multiple tools