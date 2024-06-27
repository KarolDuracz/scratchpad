// Import workers
import KeyEventWorker from './keyEventWorker.js';
import InputProcessingWorker from './inputProcessingWorker.js';

// Create workers
const keyEventWorker = new Worker('keyEventWorker.js', { type: 'module' });
const inputProcessingWorker = new Worker('inputProcessingWorker.js', { type: 'module' });

// Handle key events
document.addEventListener('keydown', (event) => {
    keyEventWorker.postMessage({ type: 'keydown', key: event.key });
});

document.addEventListener('keyup', (event) => {
    keyEventWorker.postMessage({ type: 'keyup', key: event.key });
});

// Handle input events
const inputElement = document.getElementById('userInput');
inputElement.addEventListener('input', (event) => {
    inputProcessingWorker.postMessage({ type: 'input', data: event.target.value });
});

// Listen for messages from the workers
keyEventWorker.onmessage = (event) => {
    console.log(`Key Event Worker: ${event.data}`);
};

inputProcessingWorker.onmessage = (event) => {
    console.log(`Input Processing Worker: ${event.data}`);
};

// Handle worker errors
keyEventWorker.onerror = (err) => {
    console.error(`Key Event Worker Error: ${err.message}`);
};

inputProcessingWorker.onerror = (err) => {
    console.error(`Input Processing Worker Error: ${err.message}`);
};
