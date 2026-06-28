import {JobScheduler} from './scheduler.js';
import {RequestManager} from './requestManager.js';
import {OnDeviceService} from './services/onDeviceService.js';
import {CloudService} from './services/cloudService.js';
import {Evaluator} from './evaluator.js';
import {getNumberOfWords, logTo, sleep} from './utils.js';


// get references to html elements
const logEl = document.getElementById('log-table-body');
const statsEl = document.getElementById('stats');
const deviceStatusEl = document.getElementById('deviceStatus');


// instantiate services and components
const onDeviceInferenceService = new OnDeviceService(getModelSelection());
const cloudInferenceService = new CloudService({
    apiKey: document.getElementById('cloudApiKey').value,
    model: document.getElementById('cloudModel').value
});
const evaluator = new Evaluator();


const requestManager = new RequestManager({
    deviceService: onDeviceInferenceService,
    cloudService: cloudInferenceService,
    evaluator,
    logger: evt => {
        logTo(logEl, evt);
        updateStats();
    }
});

// expose for debugging and UI updates, and listen for posterior updates
window.requestManager = requestManager;

window.addEventListener('perfModelsUpdated', (ev) => {
    const {device, cloud} = ev.detail || {};
    const el = id => document.getElementById(id);
    if (cloud) {
        if (el('cloudSlope')) el('cloudSlope').value = Number(cloud.slope ?? 0).toFixed(3);
        if (el('cloudIntercept')) el('cloudIntercept').value = Number(cloud.intercept ?? 0).toFixed(1);
    }
    if (device) {
        if (el('deviceSlope')) el('deviceSlope').value = Number(device.slope ?? 0).toFixed(3);
        if (el('deviceIntercept')) el('deviceIntercept').value = Number(device.intercept ?? 0).toFixed(1);
    }
    if (typeof updateStats === 'function') {
        try {
            updateStats();
        } catch (e) { /* ignore */
        }
    }
});


// instantiate the job scheduler
const datasetName = document.getElementById('dataset').value;
const scheduler = new JobScheduler(datasetName);


scheduler.onJob(async (job) => {
    await requestManager.pushJob(job);
});


// add event listeners for configuration inputs
document.getElementById('dataset').addEventListener('change', (e) => {
        scheduler.setDatasetName(e.target.value);
        scheduler.reloadDataset();
    }
);
document.getElementById('deviceModel').addEventListener('change', (e) => {
        onDeviceInferenceService.updateConfig(getModelSelection())
    }
);
document.getElementById('cloudModel').addEventListener('change', (e) =>
    cloudInferenceService.updateConfig({model: e.target.value})
);
document.getElementById('cloudApiKey').addEventListener('input', (e) =>
    cloudInferenceService.updateConfig({apiKey: e.target.value})
);

// add event listener for run button
document.getElementById('startBtn').addEventListener('click', async () => {

    // toggle start and stop buttons
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;

    // get configuration values from UI
    const pattern = document.getElementById('patternSelect').value;
    const routeStrategy = document.getElementById('routeStrategy').value;
    const cloudProb = parseFloat(document.getElementById('cloudProb').value);
    const devicePerfModel = {
        slope: parseFloat(document.getElementById('deviceSlope').value),
        intercept: parseFloat(document.getElementById('deviceIntercept').value)
    };
    const cloudPerfModel = {
        slope: parseFloat(document.getElementById('cloudSlope').value),
        intercept: parseFloat(document.getElementById('cloudIntercept').value)
    };

    // update request manager routing strategy
    requestManager.updateRouting({routeStrategy, cloudProb, devicePerfModel, cloudPerfModel});

    // starting is only available when model is loaded
    if (routeStrategy !== 'always_cloud' && !onDeviceInferenceService.isReady()) {
        await loadDeviceModel();
    }

    // start the job scheduler with the selected pattern
    scheduler.startPattern(pattern);
});


document.getElementById('stopBtn').addEventListener('click', () => {
    scheduler.stop();
    isExperimentRunning = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
});

document.getElementById('downloadStatsCsv').addEventListener('click', () => {
    downloadStatsAsCSV();
});
document.getElementById('loadDeviceModelBtn').addEventListener('click', () => {
    loadDeviceModel();
});

document.getElementById('interArrivalTimeLambda').addEventListener('input', (event) => {
    const newValue = parseFloat(event.target.value);
    if (!isNaN(newValue) && newValue > 0) {
        scheduler._interArrivalTimeLambda = newValue;
    }
});

let currentExperiment = null;
let experimentJobCount = 0;
let experimentTargetJobs = 0;
let isExperimentRunning = false;
const TARGET_JOBS = 500;

document.getElementById('start1000Btn').addEventListener('click', async () => {

    // Get configuration from UI
    const datasetName = document.getElementById('dataset').value;
    const pattern = document.getElementById('patternSelect').value;
    const routeStrategy = document.getElementById('routeStrategy').value;
    const cloudProb = parseFloat(document.getElementById('cloudProb').value);
    const deviceModel = getModelSelection().modelName;
    const cloudModel = document.getElementById('cloudModel').value;

    // Validate
    if (routeStrategy !== 'always_cloud' && !onDeviceInferenceService.isReady()) {
        alert('Please load the on-device model first, or select "Always Cloud" strategy.');
        return;
    }

    if (routeStrategy !== 'always_device') {
        const apiKey = document.getElementById('cloudApiKey').value;
        if (!apiKey || apiKey.trim() === '') {
            alert('Please enter a Cloud API Key, or select "Always Device" strategy.');
            return;
        }
    }

    // Store experiment config
    currentExperiment = {
        deviceModel,
        cloudModel,
        datasetName,
        routeStrategy,
        pattern,
        startTime: Date.now()
    };

    experimentJobCount = 0;
    experimentTargetJobs = TARGET_JOBS;
    isExperimentRunning = true;

    // Reset stats
    requestManager.stats.count = 0;
    requestManager.stats.cloud = 0;
    requestManager.stats.device = 0;
    requestManager.stats.totalLatencyMs = 0;
    requestManager.stats.results = [];

    // Update UI
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('start1000Btn').disabled = true;
    document.getElementById('start1000Btn').textContent = `Running`;

    // Update routing
    requestManager.updateRouting({routeStrategy, cloudProb});

    console.log(`🚀 Starting experiment: ${TARGET_JOBS} jobs`);
    console.log(`📊 Config: Strategy=${routeStrategy}, Pattern=${pattern}`);

    try {
        // Reload dataset to ensure we have enough items
        await scheduler.reloadDataset();

        // Start the limited run
        await scheduler.startPattern(pattern, TARGET_JOBS);

    } catch (error) {
        console.error('❌ Experiment error:', error);
        alert(`Experiment failed: ${error.message}`);
    }

    // wait for all jobs to complete and check every 2 seconds (yes is a bit hacky)
    // this is necessary since the jobs are processed async and some may still be running although no more jobs are scheduled
    while (isExperimentRunning && requestManager.stats.count < TARGET_JOBS) {
        await sleep(2000);
    }

    // Then finish experiment
    finishExperiment();
});

function finishExperiment() {
    isExperimentRunning = false;
    console.log('✅ Experiment complete!');

    // Stop the scheduler
    scheduler.stop();

    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('start1000Btn').disabled = false;
    document.getElementById('start1000Btn').textContent = 'Start 500';

    // Auto-download results
    setTimeout(() => {
        downloadExperimentResults();
    }, 500);
}

function downloadExperimentResults() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    // Build model name for filename
    let modelName = '';
    if (currentExperiment.routeStrategy === 'always_cloud') {
        modelName = currentExperiment.cloudModel.replace(/[^a-zA-Z0-9]/g, '-');
    } else if (currentExperiment.routeStrategy === 'always_device') {
        modelName = currentExperiment.deviceModel.split('/').pop().replace(/[^a-zA-Z0-9]/g, '-');
    } else {
        const device = currentExperiment.deviceModel.split('/').pop().replace(/[^a-zA-Z0-9]/g, '-');
        const cloud = currentExperiment.cloudModel.replace(/[^a-zA-Z0-9]/g, '-');
        modelName = `${device}_${cloud}`;
    }

    // Build stats object with experiment info
    const stats = {
        experiment: {
            ...currentExperiment,
            endTime: Date.now(),
            completedJobs: requestManager.stats.count
        },
        stats: requestManager.stats
    };

    // Download CSV files of statistics and raw results
    const filesToDownload = [
        {
            "name": `stats_experiment_${modelName}_${currentExperiment.routeStrategy}_${currentExperiment.pattern}_${timestamp}`,
            "csv": buildStatisticCSV(stats)
        },
        {
            "name": `raw_experiment_${modelName}_${currentExperiment.routeStrategy}_${currentExperiment.pattern}_${timestamp}`,
            "csv": buildExperimentCSV(stats)
        }
    ];

    for (const file of filesToDownload) {
        const csvBlob = new Blob([file.csv], {type: 'text/csv'});
        const csvUrl = URL.createObjectURL(csvBlob);
        const csvLink = document.createElement('a');
        csvLink.href = csvUrl;
        csvLink.download = `${file.name}.csv`;
        csvLink.click();
        URL.revokeObjectURL(csvUrl);

        console.log(`📥 Downloaded: ${file.name}.csv`);
    }
}

function buildExperimentCSV(stats) {
    const lines = [];

    // Header
    lines.push('dataset_item_id,route,latency_ms,total_latency_ms,queueing_time_ms,inference_time_ms,exact_match,ground_truth,answer,job_start_ts,inference_start_ts,inference_end_ts,prompt,number_of_words,number_of_characters,experiment_start_time_ms,experiment_end_time_ms,dataset_name,route_strategy,pattern,device_model,cloud_model,jseq_predicted_device_inference_time,jseq_predicted_device_total_time,jseq_expected_device_finish_time,jseq_expected_device_number_jobs,jseq_predicted_cloud_inference_time,jseq_predicted_cloud_total_time,jseq_expected_cloud_finish_time,jseq_expected_cloud_number_jobs');


    // Data rows
    stats.stats.results.forEach((result, index) => {
        const row = [
            result.job.id,
            result.route || '',
            (result.latency || 0).toFixed(2),
            (result.totalLatency || 0).toFixed(2),
            (result.queueingTime || 0).toFixed(2),
            (result.inferenceTime || 0).toFixed(2),
            result.evalRes?.exactMatch,
            `"${(result.job?.groundTruth || '').replace(/"/g, '""')}"`,
            `"${(result.text?.answer || '').replace(/"/g, '""')}"`,
            result.job.timestamps.jobStart || 0,
            result.job.timestamps.inferenceStart || 0,
            result.job.timestamps.inferenceEnd || 0,
            `"${(result.job.prompt || '').replace(/"/g, '""')}"`,
            getNumberOfWords(result.job.prompt || ''),
            result.job.prompt.length,
            stats.experiment.startTime || 0,
            stats.experiment.endTime || 0,
            stats.experiment.datasetName,
            stats.experiment.routeStrategy,
            stats.experiment.pattern,
            stats.experiment.deviceModel,
            stats.experiment.cloudModel,
            result.job?.hero_predictions?.device?.predictedInferenceTime || -1,
            result.job?.hero_predictions?.device?.predictedTotalTime || -1,
            result.job?.hero_predictions?.device?.expectedFinishTime || -1,
            result.job?.hero_predictions?.device?.numberOfJobsInQueue || 0,
            result.job?.hero_predictions?.cloud?.predictedInferenceTime || -1,
            result.job?.hero_predictions?.cloud?.predictedTotalTime || -1,
            result.job?.hero_predictions?.cloud?.expectedFinishTime || -1,
            result.job?.hero_predictions?.cloud?.numberOfJobsInQueue || 0
        ];
        lines.push(row.join(','));
    });
    return lines.join('\n');
}

/**
 * Build a CSV with the statistics of an experiment run.
 * Stores the following fields for on device, in cloud and overall:
 * - total_requests
 * - accuracy_percent
 * - avg_latency_ms
 * - avg_total_latency_ms
 * - avg_queueing_time_ms
 * - avg_inference_time_ms
 *
 * @param stats
 */
function buildStatisticCSV(stats) {
    const lines = [];

    // Header
    lines.push('route, total_requests, accuracy_percent, avg_latency_ms, avg_total_latency_ms, avg_queueing_time_ms, avg_inference_time_ms');

    // Calculate averages
    const results = stats.stats.results;
    const count = results.length;

    if (count > 0) {

        // Overall stats
        const avgLatency = results.reduce((sum, r) => sum + (r.latency || 0), 0) / count;
        const avgTotalLatency = results.reduce((sum, r) => sum + (r.totalLatency || 0), 0) / count;
        const avgQueueingTime = results.reduce((sum, r) => sum + (r.queueingTime || 0), 0) / count;
        const avgInferenceTime = results.reduce((sum, r) => sum + (r.inferenceTime || 0), 0) / count;
        const accuracy = results.filter(r => r.evalRes?.exactMatch).length / count * 100;
        lines.push(`overall, ${count}, ${accuracy.toFixed(2)}, ${avgLatency.toFixed(2)}, ${avgTotalLatency.toFixed(2)}, ${avgQueueingTime.toFixed(2)}, ${avgInferenceTime.toFixed(2)}`);

        // Device stats
        const deviceResults = results.filter(r => r.route === 'device');
        if (deviceResults.length > 0) {
            const deviceCount = deviceResults.length;
            const deviceAvgLatency = deviceResults.reduce((sum, r) => sum + (r.latency || 0), 0) / deviceCount;
            const deviceAvgTotalLatency = deviceResults.reduce((sum, r) => sum + (r.totalLatency || 0), 0) / deviceCount;
            const deviceAvgQueueingTime = deviceResults.reduce((sum, r) => sum + (r.queueingTime || 0), 0) / deviceCount;
            const deviceAvgInferenceTime = deviceResults.reduce((sum, r) => sum + (r.inferenceTime || 0), 0) / deviceCount;
            const deviceAccuracy = deviceResults.filter(r => r.evalRes?.exactMatch).length / deviceCount * 100;
            lines.push(`device, ${deviceCount}, ${deviceAccuracy.toFixed(2)}, ${deviceAvgLatency.toFixed(2)}, ${deviceAvgTotalLatency.toFixed(2)}, ${deviceAvgQueueingTime.toFixed(2)}, ${deviceAvgInferenceTime.toFixed(2)}`);
        } else {
            lines.push(`device, 0, 0.00, 0.00, 0.00, 0.00, 0.00`);
        }

        // Cloud stats
        const cloudResults = results.filter(r => r.route === 'cloud');
        if (cloudResults.length > 0) {
            const cloudCount = cloudResults.length;
            const cloudAvgLatency = cloudResults.reduce((sum, r) => sum + (r.latency || 0), 0) / cloudCount;
            const cloudAvgTotalLatency = cloudResults.reduce((sum, r) => sum + (r.totalLatency || 0), 0) / cloudCount;
            const cloudAvgQueueingTime = cloudResults.reduce((sum, r) => sum + (r.queueingTime || 0), 0) / cloudCount;
            const cloudAvgInferenceTime = cloudResults.reduce((sum, r) => sum + (r.inferenceTime || 0), 0) / cloudCount;
            const cloudAccuracy = cloudResults.filter(r => r.evalRes?.exactMatch).length / cloudCount * 100;
            lines.push(`cloud, ${cloudCount}, ${cloudAccuracy.toFixed(2)}, ${cloudAvgLatency.toFixed(2)}, ${cloudAvgTotalLatency.toFixed(2)}, ${cloudAvgQueueingTime.toFixed(2)}, ${cloudAvgInferenceTime.toFixed(2)}`);
        } else {
            lines.push(`cloud, 0, 0.00, 0.00, 0.00, 0.00, 0.00`);
        }
    }

    return lines.join('\n');
}

function getModelSelection() {
    try {
        return JSON.parse(document.getElementById('deviceModel').value);
    } catch (error) {
        console.error('Invalid JSON in model selection:', value);
        return null;
    }
}


async function loadDeviceModel() {
    deviceStatusEl.textContent = 'Loading...';
    document.getElementById('loadDeviceModelBtn').disabled = true;
    document.getElementById('loadDeviceModelBtn').textContent = 'Loading Model...';
    const loadingBar = document.getElementById('deviceLoadingBar');
    const loadingText = document.getElementById('deviceLoadingText');
    loadingBar.style.width = '0%';
    loadingText.textContent = '';
    const files = {};

    function updateModelLoadingUI(progress) {
        console.log('Model loading progress:', progress);
        if (progress && progress.loaded && progress.total) {
            files[progress.file] = {loaded: progress.loaded, total: progress.total};
            const fileNames = Object.keys(files);
            const hasOnnxFile = Boolean(fileNames.find(name => name.endsWith('.onnx')));
            if (!hasOnnxFile) {
                loadingBar.style.width = '0%';
                loadingText.textContent = `Loading: 0% (0 GB / ... GB)`;
                return;
            }
            const filesArray = Object.values(files);
            const totalBytes = filesArray.reduce((total, file) => total + file.total, 0);
            const loadedBytes = filesArray.reduce((total, file) => total + file.loaded, 0);
            const percent = ((loadedBytes / totalBytes) * 100).toFixed(1);
            loadingBar.style.width = percent + '%';
            loadingText.textContent = `Loading: ${percent}% (${(loadedBytes / (1024 ** 3)).toFixed(2)} GB / ${(totalBytes / (1024 ** 3)).toFixed(2)} GB)`;
        } else if (progress && progress.status) {
            loadingText.textContent = progress.status;
        } else if (typeof progress === 'string') {
            loadingText.textContent = progress;
        }
    }

    try {
        await onDeviceInferenceService.load(updateModelLoadingUI);
        deviceStatusEl.textContent = 'Model Ready';
        loadingBar.style.width = '100%';
        loadingText.textContent = 'Model loaded.';
        document.getElementById('loadDeviceModelBtn').disabled = false;
        document.getElementById('loadDeviceModelBtn').textContent = 'Load Model';

    } catch (e) {
        console.error('❌ Error loading on-device model:', e);
        deviceStatusEl.textContent = `Error: ${e.message}`;
        loadingText.textContent = 'Error loading model.';
        document.getElementById('loadDeviceModelBtn').disabled = false;
        document.getElementById('loadDeviceModelBtn').textContent = 'Load Model';
    }
}

function downloadStatsAsCSV() {
    // make the stats compatible with buildExperimentCSV method for reuse
    const stats = {
        experiment: {
            deviceModel: getModelSelection().modelName,
            cloudModel: document.getElementById('cloudModel').value,
            datasetName: document.getElementById('dataset').value,
            routeStrategy: document.getElementById('routeStrategy').value,
            pattern: document.getElementById('patternSelect').value,
            startTime: null,
            endTime: Date.now(),
            completedJobs: requestManager.stats.count
        },
        stats: requestManager.stats
    };

    const csvContent = buildExperimentCSV(stats);

    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csvContent);
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "stats.csv");
    dlAnchorElem.click();
}

/**
 * Update the statistics display in the UI based on the request manager's stats
 */
function updateStats() {
    const s = requestManager.stats;

    // Calculate average timing metrics
    const avgTotalLatency = s.count ? (s.results.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.count) : 0;
    const avgQueueingTime = s.count ? (s.results.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.count) : 0;
    const avgInferenceTime = s.count ? (s.results.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.count) : 0;

    const cloudResults = s.results.filter(e => e.route === 'cloud');
    const deviceResults = s.results.filter(e => e.route === 'device');

    const avgCloudTotal = s.cloud ? (cloudResults.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.cloud) : 0;
    const avgCloudQueue = s.cloud ? (cloudResults.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.cloud) : 0;
    const avgCloudInference = s.cloud ? (cloudResults.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.cloud) : 0;

    const avgDeviceTotal = s.device ? (deviceResults.reduce((a, b) => a + (b.totalLatency || 0), 0) / s.device) : 0;
    const avgDeviceQueue = s.device ? (deviceResults.reduce((a, b) => a + (b.queueingTime || 0), 0) / s.device) : 0;
    const avgDeviceInference = s.device ? (deviceResults.reduce((a, b) => a + (b.inferenceTime || 0), 0) / s.device) : 0;

    statsEl.innerHTML = `
        <div style="display: flex; justify-content: space-between;">
            <div>
                <h3>General Stats</h3>
                <pre>
Requests: ${s.count}
Avg total latency: ${avgTotalLatency.toFixed(1)}ms
Avg queueing time: ${avgQueueingTime.toFixed(1)}ms
Avg inference time: ${avgInferenceTime.toFixed(1)}ms
Avg correct: ${s.count ? (s.results.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.count * 100).toFixed(1) : 0}%
                </pre>
            </div>
            <div>
                <h3>Cloud Stats</h3>
                <pre>
Requests: ${s.cloud}
Avg total latency: ${avgCloudTotal.toFixed(1)}ms
Avg queueing time: ${avgCloudQueue.toFixed(1)}ms
Avg inference time: ${avgCloudInference.toFixed(1)}ms
Avg correct: ${s.cloud ? (cloudResults.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.cloud * 100).toFixed(1) : 0}%
               
                </pre>
            </div>
            <div>
                <h3>On-Device Stats</h3>
                <pre>
Requests: ${s.device}
Avg total latency: ${avgDeviceTotal.toFixed(1)}ms
Avg queueing time: ${avgDeviceQueue.toFixed(1)}ms
Avg inference time: ${avgDeviceInference.toFixed(1)}ms
Avg correct: ${s.device ? (deviceResults.reduce((a, b) => a + (b.evalRes.exactMatch ? 1 : 0), 0) / s.device * 100).toFixed(1) : 0}%

                </pre>
            </div>
        </div>`;
}