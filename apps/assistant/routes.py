from flask import Flask, render_template, request, jsonify, session
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import os

from apps.assistant import blueprint

# Load environment variables

groq_api_key = os.environ['GROQ_API_KEY']


# Initialize Groq Langchain chat object
def get_conversation_chain(model_name):
    memory = ConversationBufferWindowMemory(k=5)  # Default length of 5, can be adjusted
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    return ConversationChain(llm=groq_chat, memory=memory)

# Limit the response to 120 words and format it in bullet points
def format_user_prompt(question):
    formatted_prompt = f"""
    {question}

    Please provide the answer , and the total response should not exceed 30 words.
    """
    return formatted_prompt



def process_response(response_text):
    # Split the response by lines or periods for numbered format
    points = response_text.split('\n')  # Assuming each numbered point comes in a new line
    
    # Clean and reformat the response to remove any additional symbols and ensure numbered format
    formatted_response = ''
    for i, point in enumerate(points):
        point = point.strip('- ')  # Strip any unwanted symbols like dashes or extra spaces
        if point:
            formatted_response += f"{i+1}. {point}<br>"  # Add HTML line break and numbering

    return formatted_response


@blueprint.route('/aiasiss', methods=['GET', 'POST'])
def aiassis():
    if request.method == 'POST':
        user_question = request.form['question']
        selected_model = request.form['model']
        memory_length = int(request.form['memory_length'])

        # Update memory length in ConversationChain
        conversation_chain = get_conversation_chain(selected_model)
        conversation_chain.memory.k = memory_length

        # Format the question with the word limit and point format instructions
        formatted_prompt = format_user_prompt(user_question)

        # Process the user question
        response = conversation_chain.invoke(formatted_prompt)

        # Post-process response to include numbered points and line breaks
        ai_response = process_response(response['response'])

        # Update the chat history
        chat_history = session.get('chat_history', [])
        chat_history.append({'human': user_question, 'AI': ai_response})
        session['chat_history'] = chat_history
        
        return render_template('home/aiassis.html', response=ai_response, chat_history=chat_history)
    
    return render_template('home/aiassis.html', segment="aiassis", response=None, chat_history=session.get('chat_history', []), show_sideBar=True)


@blueprint.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('message')
    selected_model = "mixtral-8x7b-32768"  # Example model, adjust as needed
    memory_length = 5  # Default memory length

    # Get conversation chain and process the user's message
    conversation_chain = get_conversation_chain(selected_model)
    conversation_chain.memory.k = memory_length

    # Format the question with word limit and bullet points
    formatted_prompt = format_user_prompt(user_question)
    response = conversation_chain.invoke(formatted_prompt)

    # Update the chat history stored in the session
    chat_history = session.get('chat_history', [])
    chat_history.append({'human': user_question, 'AI': response['response']})
    session['chat_history'] = chat_history

    return jsonify({'response': response['response']})


