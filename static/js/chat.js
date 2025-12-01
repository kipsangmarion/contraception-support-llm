// Chat application logic

const chatContainer = document.getElementById('chat-container');
const messageForm = document.getElementById('message-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const languageSelect = document.getElementById('language');
const typingIndicator = document.getElementById('typing-indicator');
const errorMessage = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

// API endpoint
const API_URL = '/counsel/query';

// Session ID for conversation continuity
let sessionId = generateSessionId();
let userId = generateUserId();

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Generate unique user ID
function generateUserId() {
    let stored = localStorage.getItem('user_id');
    if (!stored) {
        stored = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('user_id', stored);
    }
    return stored;
}

// Add message to chat
function addMessage(content, isUser = false, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex items-start space-x-3 message';

    if (isUser) {
        // User message
        messageDiv.innerHTML = `
            <div class="user-message ml-auto">
                <p>${escapeHtml(content)}</p>
            </div>
            <div class="bg-gradient-to-r from-purple-500 to-pink-500 rounded-full p-2 flex-shrink-0">
                <i class="fas fa-user text-white"></i>
            </div>
        `;
    } else {
        // AI message
        const sourcesHtml = sources.length > 0 ? `
            <div class="source-citation mt-2">
                <p class="font-semibold text-xs text-gray-600 mb-1">📚 Sources:</p>
                ${sources.map((s, i) => `
                    <p class="text-xs text-gray-600">
                        ${i + 1}. ${s.source} (Page ${s.page})
                    </p>
                `).join('')}
            </div>
        ` : '';

        messageDiv.innerHTML = `
            <div class="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full p-2 flex-shrink-0">
                <i class="fas fa-robot text-white"></i>
            </div>
            <div class="ai-message">
                <p class="whitespace-pre-wrap">${escapeHtml(content)}</p>
                ${sourcesHtml}
            </div>
        `;
    }

    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Show/hide typing indicator
function showTyping(show = true) {
    if (show) {
        typingIndicator.classList.remove('hidden');
    } else {
        typingIndicator.classList.add('hidden');
    }
}

// Show/hide error
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
    setTimeout(() => {
        errorMessage.classList.add('hidden');
    }, 5000);
}

// Scroll to bottom
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Send message to API
async function sendMessage(message) {
    const language = languageSelect.value;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                language: language,
                session_id: sessionId,
                user_id: userId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;

    } catch (error) {
        console.error('Error sending message:', error);
        throw error;
    }
}

// Handle form submission
messageForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const message = userInput.value.trim();
    if (!message) return;

    // Disable input while processing
    userInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to chat
    addMessage(message, true);

    // Clear input
    userInput.value = '';

    // Show typing indicator
    showTyping(true);

    try {
        // Send to API
        const response = await sendMessage(message);

        // Hide typing indicator
        showTyping(false);

        // Add AI response
        addMessage(response.response, false, response.sources || []);

    } catch (error) {
        showTyping(false);
        showError('Failed to get response. Please check if the server is running and try again.');
        console.error('Error:', error);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
});

// Handle Enter key (but allow Shift+Enter for newlines)
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        messageForm.dispatchEvent(new Event('submit'));
    }
});

// Language change notification
languageSelect.addEventListener('change', () => {
    const language = languageSelect.value;
    const langNames = {
        'english': 'English',
        'french': 'Français',
        'kinyarwanda': 'Kinyarwanda'
    };

    addMessage(`Language changed to ${langNames[language]}. You can now ask questions in ${langNames[language]}.`, false);
});

// Focus input on load
window.addEventListener('load', () => {
    userInput.focus();
});

// Example questions for quick testing
const exampleQuestions = [
    "What is the effectiveness of the copper IUD?",
    "How do birth control pills work?",
    "What are the side effects of DMPA injections?",
    "Can I use emergency contraception?",
    "When does fertility return after stopping the implant?"
];

// Optional: Add quick question buttons (can be added to HTML if desired)
function addQuickQuestions() {
    const quickQuestionsDiv = document.createElement('div');
    quickQuestionsDiv.className = 'mt-4 space-y-2';
    quickQuestionsDiv.innerHTML = '<p class="text-sm text-gray-600 font-semibold">Try asking:</p>';

    exampleQuestions.slice(0, 3).forEach(question => {
        const btn = document.createElement('button');
        btn.className = 'block w-full text-left px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors';
        btn.textContent = question;
        btn.addEventListener('click', () => {
            userInput.value = question;
            messageForm.dispatchEvent(new Event('submit'));
        });
        quickQuestionsDiv.appendChild(btn);
    });

    // Can be added after welcome message
    // chatContainer.appendChild(quickQuestionsDiv);
}

console.log('✅ Chat interface initialized');
console.log(`Session ID: ${sessionId}`);
console.log(`User ID: ${userId}`);
