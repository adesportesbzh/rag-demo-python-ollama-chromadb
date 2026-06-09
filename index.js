document.addEventListener("DOMContentLoaded", () => {

const messagesDiv = document.getElementById("messages");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

function addMessage(text, sender) {
    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.textContent = text;
    messagesDiv.appendChild(msg);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return msg;
}

function setLoading(isLoading) {
    input.disabled = isLoading;
    sendBtn.disabled = isLoading;
}

async function sendQuestion() {
    const question = input.value.trim();
    if (!question) return;

    addMessage(question, "user");
    input.value = "";
    setLoading(true);

    const loadingMsg = addMessage("Thinking…", "loading");

    try {
        const res = await fetch(`/answer_question?q=${encodeURIComponent(question)}`);
        const data = await res.json();

        loadingMsg.remove();
        addMessage(data.answer, "agent");
    } catch (err) {
        loadingMsg.remove();
        addMessage("Error contacting server.", "agent");
    } finally {
        setLoading(false);
    }
}

sendBtn.addEventListener("click", sendQuestion);

input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        sendQuestion();
    }
});

});
