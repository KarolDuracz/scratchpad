const { parentPort } = require('worker_threads');

parentPort.on('message', (event) => {
    if (event.type === 'input') {
        const processedInput = event.data.toUpperCase();
        parentPort.postMessage(`Processed input: ${processedInput}`);
    }
});
