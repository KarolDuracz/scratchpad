const { parentPort } = require('worker_threads');

parentPort.on('message', (event) => {
    if (event.type === 'keydown') {
        parentPort.postMessage(`Key down: ${event.key.name}`);
    } else if (event.type === 'keyup') {
        parentPort.postMessage(`Key up: ${event.key.name}`);
    }
});
