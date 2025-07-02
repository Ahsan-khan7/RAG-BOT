document.addEventListener('DOMContentLoaded', () => {
    const pdfFileInput = document.getElementById('pdfFileInput');
    const uploadPdfBtn = document.getElementById('uploadPdfBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    const chatWindow = document.getElementById('chatWindow');
    const chatInput = document.getElementById('chatInput');
    const sendMessageBtn = document.getElementById('sendMessageBtn');

    const API_BASE_URL = 'http://127.0.0.1:5000'; // Flask backend URL

    // Function to add a message to the chat window
    function addMessage(sender, message) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('p-3', 'rounded-lg', 'mb-2', 'shadow-sm');
        if (sender === 'user') {
            messageDiv.classList.add('bg-blue-500', 'text-white', 'self-end', 'rounded-br-none', 'ml-auto', 'user-message');
        } else {
            messageDiv.classList.add('bg-gray-200', 'text-gray-800', 'self-start', 'rounded-bl-none', 'bot-message');
        }
        messageDiv.textContent = message;
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to bottom
    }

    // PDF Upload Event Listener
    uploadPdfBtn.addEventListener('click', async () => {
        const file = pdfFileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a PDF file.';
            uploadStatus.className = 'text-sm mt-2 text-center text-red-600';
            return;
        }

        uploadStatus.textContent = 'Uploading and processing PDF... This may take a moment.';
        uploadStatus.className = 'text-sm mt-2 text-center text-blue-600';

        const formData = new FormData();
        formData.append('pdf_file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                uploadStatus.textContent = data.message;
                uploadStatus.className = 'text-sm mt-2 text-center text-green-600';
                chatWindow.innerHTML = '<div class="text-center text-gray-500 italic">PDF processed! You can now ask questions about the document.</div>';
            } else {
                uploadStatus.textContent = `Error: ${data.error}`;
                uploadStatus.className = 'text-sm mt-2 text-center text-red-600';
            }
        } catch (error) {
            console.error('Error uploading PDF:', error);
            uploadStatus.textContent = 'Network error or server unavailable.';
            uploadStatus.className = 'text-sm mt-2 text-center text-red-600';
        }
    });

    // Send Message Event Listener
    sendMessageBtn.addEventListener('click', async () => {
        const message = chatInput.value.trim();
        if (message === '') return;

        addMessage('user', message);
        chatInput.value = '';

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            const data = await response.json();
            if (response.ok) {
                addMessage('bot', data.response);
            } else {
                addMessage('bot', `Error: ${data.error}`);
                console.error('Chat error:', data.error);
            }
        } catch (error) {
            console.error('Network error during chat:', error);
            addMessage('bot', 'Network error or server unavailable. Please try again.');
        }
    });

    // Allow sending message with Enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessageBtn.click();
        }
    });
});
