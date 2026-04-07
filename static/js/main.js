// 初始化图表
let trainingChart, explorationChart;

// 加载数据并渲染页面
async function loadData() {
    try {
        // 加载策略
        const policyResponse = await fetch('/api/policy');
        const policy = await policyResponse.json();
        renderPolicy(policy);
        
        // 加载统计数据
        const statsResponse = await fetch('/api/stats');
        const stats = await statsResponse.json();
        renderStats(stats);
        
        // 加载训练数据
        const trainingResponse = await fetch('/api/training');
        const trainingData = await trainingResponse.json();
        renderTrainingChart(trainingData);
        
        // 加载仿真数据
        const simResponse = await fetch('/api/simulation');
        const simData = await simResponse.json();
        renderSimulationChart(simData);
        renderRewardsChart(simData);
        
        // 加载探索分析数据
        const explorationResponse = await fetch('/api/exploration');
        const explorationData = await explorationResponse.json();
        renderExplorationChart(explorationData);
        
        // 渲染策略热力图
        renderPolicyHeatmap(policy);
        
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function renderPolicy(policy) {
    const idleContainer = document.getElementById('idle-policy');
    const congestedContainer = document.getElementById('congested-policy');
    
    idleContainer.innerHTML = '';
    congestedContainer.innerHTML = '';
    
    policy.idle.forEach((action, idx) => {
        const deviceCard = createDeviceCard(idx + 1, action);
        idleContainer.appendChild(deviceCard);
    });
    
    policy.congested.forEach((action, idx) => {
        const deviceCard = createDeviceCard(idx + 1, action);
        congestedContainer.appendChild(deviceCard);
    });
}

function createDeviceCard(deviceId, action) {
    const card = document.createElement('div');
    card.className = `device-card ${action === 1 ? 'transmit' : 'wait'}`;
    card.innerHTML = `
        <div class="device-id">设备 ${deviceId}</div>
        <div class="device-action">${action === 1 ? '📡 传输' : '⏸️ 等待'}</div>
    `;
    return card;
}

function renderStats(stats) {
    document.getElementById('idle-ratio').textContent = `${stats.idle_ratio.toFixed(1)}%`;
    document.getElementById('avg-transmitting').textContent = stats.avg_transmitting.toFixed(2);
    document.getElementById('total-reward').textContent = stats.total_reward.toFixed(2);
    document.getElementById('steps').textContent = stats.steps;
}

function renderTrainingChart(data) {
    const ctx = document.getElementById('training-chart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    const datasets = [];
    data.rewards.forEach((rewards, idx) => {
        const smoothRewards = rewards.map((_, i) => {
            if (i < 10) return rewards[i];
            return rewards.slice(i-10, i+1).reduce((a,b) => a+b, 0) / 11;
        });
        
        datasets.push({
            label: `设备 ${idx + 1}`,
            data: smoothRewards,
            borderColor: `hsl(${idx * 90}, 70%, 50%)`,
            backgroundColor: `hsla(${idx * 90}, 70%, 50%, 0.1)`,
            borderWidth: 2,
            fill: false,
            tension: 0.4
        });
    });
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Episode' }
                },
                y: {
                    title: { display: true, text: 'Average Reward' }
                }
            }
        }
    });
}

function renderSimulationChart(data) {
    const states = data.map(d => d.state_code);
    const transmitting = data.map(d => d.num_transmitting);
    
    const trace1 = {
        x: data.map(d => d.step),
        y: states,
        name: 'Channel State (1=Congested)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#e74c3c', width: 2 },
        marker: { size: 4 }
    };
    
    const trace2 = {
        x: data.map(d => d.step),
        y: transmitting,
        name: 'Transmitting Devices',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3498db', width: 2 },
        marker: { size: 4 },
        yaxis: 'y2'
    };
    
    const layout = {
        title: '信道状态与传输设备数',
        xaxis: { title: 'Time Step' },
        yaxis: { title: 'Channel State', range: [-0.5, 1.5], tickvals: [0, 1], ticktext: ['Idle', 'Congested'] },
        yaxis2: { title: 'Transmitting Devices', overlaying: 'y', side: 'right', range: [-0.5, 4.5] },
        hovermode: 'closest',
        height: 400
    };
    
    Plotly.newPlot('simulation-chart', [trace1, trace2], layout);
}

function renderRewardsChart(data) {
    const deviceRewards = [[], [], [], []];
    
    data.forEach(step => {
        step.rewards.forEach((reward, idx) => {
            if (deviceRewards[idx].length === 0) {
                deviceRewards[idx].push(reward);
            } else {
                deviceRewards[idx].push(deviceRewards[idx][deviceRewards[idx].length - 1] + reward);
            }
        });
    });
    
    const traces = deviceRewards.map((rewards, idx) => ({
        x: data.map(d => d.step),
        y: rewards,
        name: `Device ${idx + 1}`,
        type: 'scatter',
        mode: 'lines',
        line: { width: 2 }
    }));
    
    const layout = {
        title: '累计收益',
        xaxis: { title: 'Time Step' },
        yaxis: { title: 'Cumulative Reward' },
        hovermode: 'closest',
        height: 400
    };
    
    Plotly.newPlot('rewards-chart', traces, layout);
}

function renderExplorationChart(data) {
    const ctx = document.getElementById('exploration-chart').getContext('2d');
    
    if (explorationChart) {
        explorationChart.destroy();
    }
    
    explorationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.epsilons.map(e => `ε = ${e}`),
            datasets: [{
                label: 'Average Reward',
                data: data.rewards,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: { callbacks: { label: (ctx) => `Reward: ${ctx.raw.toFixed(2)}` } }
            },
            scales: {
                y: {
                    title: { display: true, text: 'Average Reward' },
                    beginAtZero: true
                },
                x: {
                    title: { display: true, text: 'Exploration Rate (ε)' }
                }
            }
        }
    });
}

function renderPolicyHeatmap(policy) {
    const data = [
        policy.idle.map(a => a === 1 ? '传输' : '等待'),
        policy.congested.map(a => a === 1 ? '传输' : '等待')
    ];
    
    const trace = {
        z: [[...policy.idle], [...policy.congested]],
        x: ['设备1', '设备2', '设备3', '设备4'],
        y: ['空闲', '拥堵'],
        type: 'heatmap',
        colorscale: 'YlOrRd',
        showscale: true,
        text: data,
        texttemplate: '%{text}',
        textfont: { size: 12 }
    };
    
    const layout = {
        title: '策略热力图',
        xaxis: { title: '设备' },
        yaxis: { title: '信道状态' },
        height: 300
    };
    
    Plotly.newPlot('policy-heatmap', [trace], layout);
}

// 页面加载时执行
document.addEventListener('DOMContentLoaded', loadData);

// 自动刷新数据（每10秒）
setInterval(loadData, 10000);