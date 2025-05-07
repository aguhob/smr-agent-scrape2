import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import faiss
import smtplib
from fpdf import FPDF
from email.message import EmailMessage
from openai import OpenAI

# Fix Unicode issues for PDF export
def clean_text(text):
    return text.encode("latin1", "replace").decode("latin1")

# Load the scraped nuclear sources
df_sources = pd.read_csv("scraped_smr_sources.csv")
embedding_model = "text-embedding-ada-002"

@st.cache_resource
def embed_sources_and_build_index():
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    texts = df_sources["preview"].astype(str).tolist()
    embeddings = []
    for i, text in enumerate(texts):
        try:
            response = client.embeddings.create(model=embedding_model, input=text)
            embeddings.append(np.array(response.data[0].embedding, dtype="float32"))
        except Exception as e:
            st.warning(f"Embedding failed for chunk {i}: {e}")
            embeddings.append(np.zeros(1536, dtype="float32"))
    index = faiss.IndexFlatL2(1536)
    index.add(np.array(embeddings))
    return index, texts

faiss_index, nuclear_chunks = embed_sources_and_build_index()

def retrieve_relevant_chunks(user_query, k=3):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.embeddings.create(model=embedding_model, input=user_query)
        query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, k)
        return [nuclear_chunks[i] for i in indices[0] if i < len(nuclear_chunks)]
    except Exception as e:
        st.warning(f"Retrieval error: {e}")
        return []

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
EMAIL_SENDER = st.secrets["EMAIL_SENDER"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
EMAIL_RECIPIENT = st.secrets["EMAIL_RECIPIENT"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

st.title("Infrastructure AI Agent Pipeline")

project_name = st.text_input("Project Name")
location = st.text_input("Location")

power_type = st.multiselect("Type of Power Generation Being Considered", [
    "Small Modular Reactor",
    "Large Giga-scale Reactor",
    "Natural Gas Turbine",
    "Wind Farm",
    "Solar Photovoltaic Farm",
    "Hybrid -- Wind/Solar+LDES",
    "Hybrid -- SMR+Natural Gas",
    "Geothermal"
])

infra_type = st.multiselect("Type of Infrastructure Being Considered", [
    "Data Center Power",
    "Industrial Heat and Power",
    "Grid Power",
    "Carbon Capture Storage + Utilization Power",
    "Remote Community Power",
    "Advanced Low-Carbon Fuels + Materials"
])

strategic_objectives = st.multiselect("Strategic Objectives Being Considered", [
    "Enhance Energy Security + Safety Systems",
    "Improve Fuel Utilization + Waste Reduction",
    "Build a Skilled Workforce",
    "Build Public Trust",
    "Reduce Reliance on Fossil Fuels + Support Decarbonization Goals",
    "Reduce Construction + Operating Costs",
    "Streamline + Align Construction Processes"
])

anticipated_risks = st.multiselect("What Risks Do You Anticipate?", [
    "High Construction Costs",
    "Long Project Timelines",
    "Safety Concerns",
    "Competition from Cheaper Energy Sources",
    "Complex + Inadequate Regulatory Framework",
    "Significant Upfront Capital Investment"
])

timeline_constraints = st.multiselect("Desired Timeline or Constraints Being Considered", [
    "Prototypes & Proof of Concept",
    "Memorandums of Understanding",
    "Power Purchase Agreements",
    "Regulator Engagement",
    "Construction & Development",
    "Other…"
])

known_partners = st.text_input("Who are Some Known Developers, Vendors or Partners")
user_name = st.text_input("Your Name")
user_email = st.text_input("Your Contact Email")

if st.button("Run Full Agent Analysis"):
    with st.spinner("Running Agent 1: Strategic Recommendation..."):
        agent1_chunks = retrieve_relevant_chunks(f"{project_name} {location} {', '.join(power_type)}")
        agent1_prompt = f"""
You are Agent 1, a strategic infrastructure advisor.
Context from recent nuclear-related sources:
{"".join(['- ' + chunk + '\\n' for chunk in agent1_chunks])}

Evaluate the following:
Project: {project_name} in {location}
Power Type: {', '.join(power_type)}
Infrastructure Type: {', '.join(infra_type)}
Objectives: {', '.join(strategic_objectives)}
Known Risks: {', '.join(anticipated_risks)}
Constraints: {', '.join(timeline_constraints)}
Partners: {known_partners}

Respond with:
1) Summary of risks,
2) Strategic Recommendation,
3) One-sentence rationale,
4) Rationale details.
"""
        agent1 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a strategic advisor."},
                {"role": "user", "content": agent1_prompt}
            ]
        )
        agent1_output = agent1.choices[0].message.content

    with st.spinner("Running Agent 2: Risk Identification..."):
        agent2_chunks = retrieve_relevant_chunks(f"{project_name} {', '.join(strategic_objectives)}")
        agent2_prompt = f"""
You are Agent 2, a nuclear infrastructure risk translator.
Context from recent nuclear-related sources:
{"".join(['- ' + chunk + '\\n' for chunk in agent2_chunks])}

Identify 3–5 core risks in:
Project: {project_name}, Objectives: {', '.join(strategic_objectives)}

Output format:
- Risk Type
- Description
- Urgency Level (Low, Medium, High)
"""
        agent2 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a nuclear risk analyst."},
                {"role": "user", "content": agent2_prompt}
            ]
        )
        agent2_output = agent2.choices[0].message.content

    with st.spinner("Running Agent 3: Mitigation Planning..."):
        agent3_chunks = retrieve_relevant_chunks(agent2_output)
        agent3_prompt = f"""
You are Agent 3, a mitigation planner for nuclear infrastructure projects.
Context from nuclear-specific sources:
{"".join(['- ' + chunk + '\\n' for chunk in agent3_chunks])}

Based on these risks:
{agent2_output}

Generate:
- Mitigation Strategy per risk
- Clear and actionable execution steps
"""
        agent3 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a risk mitigation strategist."},
                {"role": "user", "content": agent3_prompt}
            ]
        )
        agent3_output = agent3.choices[0].message.content

    # Generate and download PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Project Summary\nDate: {datetime.datetime.now().strftime('%Y-%m-%d')}\nProject: {project_name}\nLocation: {location}\nPower Type: {', '.join(power_type)}\nInfrastructure Type: {', '.join(infra_type)}\nObjectives: {', '.join(strategic_objectives)}\nPartners: {known_partners}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, clean_text(f"Agent 1 Output:\n{agent1_output}"))
    pdf.add_page()
    pdf.multi_cell(0, 10, clean_text(f"Agent 2 Risk Summary:\n{agent2_output}"))
    pdf.add_page()
    pdf.multi_cell(0, 10, clean_text(f"Agent 3 Mitigation Plan:\n{agent3_output}"))
    pdf_path = f"{project_name.replace(' ', '_')}_AI_Plan.pdf"
    pdf.output(pdf_path)
    st.download_button("Download PDF", file_name=pdf_path.split("/")[-1], data=open(pdf_path, "rb"), mime="application/pdf")

    # Email results
    msg = EmailMessage()
    msg["Subject"] = f"AI Project Review: {project_name}"
    msg["From"] = EMAIL_SENDER
    msg["To"] = user_email
    msg["Cc"] = EMAIL_RECIPIENT
    msg.set_content(clean_text(f"Hello {user_name},\n\nAttached is your full AI analysis for the project: {project_name}."))
    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=pdf_path.split("/")[-1])
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    st.success("PDF emailed to stakeholder!")
    st.markdown("### ✅ Submission Complete")
    st.markdown(f"Thanks, **{user_name}**! A copy of your AI-generated project review has been emailed to you and logged for internal review.")
