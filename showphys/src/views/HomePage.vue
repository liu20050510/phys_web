<template>
    <div class="container">
        <!-- 共用侧边导航栏 -->
        <el-menu class="sidebar" :default-active="activeMenu" router background-color="#304156" text-color="#bfcbd9"
            active-text-color="#409EFF">
            <div class="logo">欢迎使用本系统！</div>
            <el-menu-item index="/Welcome">
                <el-icon>
                    <House />
                </el-icon>
                <span>首页</span>
            </el-menu-item>
            <el-menu-item index="/Home">
                <el-icon>
                    <Odometer />
                </el-icon>
                <span>心率监测</span>
            </el-menu-item>
            <el-menu-item index="/History">
                <el-icon>
                    <Clock />
                </el-icon>
                <span>历史记录</span>
            </el-menu-item>
            <el-menu-item index="/About">
                <el-icon>
                    <InfoFilled />
                </el-icon>
                <span>关于我们</span>
            </el-menu-item>
        </el-menu>

        <div class="main-content">
            <!-- 新增标题栏 -->
            <div class="header-bar">
                <div class="system-title">基于多路径协同交互的远程生理监测系统</div>
                <div class="user-info">
                    <el-avatar :size="40" :icon="User" class="user-avatar" />
                    <span class="user-name">{{ username || 'User' }}</span>
                </div>
            </div>

            <!-- 实时心率检测主区域 -->
            <el-row :gutter="20">
                <el-col :span="14">
                    <el-card class="camera-card">
                        <div class="camera-container">
                            <video
                                ref="videoRef"
                                autoplay
                                playsinline
                                class="camera-feed"
                                v-show="!isProcessing"
                            ></video>
                            <canvas
                                ref="canvasRef"
                                class="camera-canvas"
                                v-show="isProcessing"
                            ></canvas>
                            <div v-if="!isCameraActive && !isProcessing" class="camera-placeholder">
                                <el-icon size="60"><VideoCamera /></el-icon>
                                <p>摄像头未激活</p>
                            </div>
                        </div>

                        <div class="control-panel">
                            <el-button
                                type="primary"
                                @click="startDetection"
                                :disabled="isDetecting || !isCameraActive"
                                v-if="!isDetecting"
                            >
                                开始检测
                            </el-button>
                            <el-button
                                type="danger"
                                @click="stopDetection"
                                :disabled="!isDetecting"
                                v-else
                            >
                                停止检测
                            </el-button>

                            <el-button @click="captureImage" :disabled="!isDetecting">
                                截图
                            </el-button>
                        </div>
                    </el-card>
                </el-col>

                <el-col :span="10">
                    <el-card class="info-card">
                        <div class="heart-rate-display">
                            <div class="heart-rate-value">
                                <span class="rate">{{ currentHeartRate || '--' }}</span>
                                <span class="unit">BPM</span>
                            </div>
                            <div class="status">
                                <el-tag :type="getStatusType()">{{ getStatusText() }}</el-tag>
                            </div>
                        </div>

                        <div class="heart-rate-chart">
                            <div ref="heartRateChart" class="chart-container"></div>
                        </div>

                        <div class="detection-info">
                            <el-descriptions :column="1" size="small" border>
                                <el-descriptions-item label="检测时间">
                                    {{ formatDateTime(detectionTime) || '未开始' }}
                                </el-descriptions-item>
                                <el-descriptions-item label="检测时长">
                                    {{ formatDuration(detectionDuration) }}
                                </el-descriptions-item>
                                <el-descriptions-item label="平均心率">
                                    {{ averageHeartRate || '--' }} BPM
                                </el-descriptions-item>
                            </el-descriptions>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'
import dayjs from 'dayjs'
import {
    House,
    Clock,
    InfoFilled,
    User,
    Odometer,
    VideoCamera
} from '@element-plus/icons-vue'
import axios from '@/apis/axios'
import { useRouter } from 'vue-router'

// 数据与状态
const activeMenu = ref('/Home')
const videoRef = ref(null)
const canvasRef = ref(null)
const isCameraActive = ref(false)
const isDetecting = ref(false)
const isProcessing = ref(false)
const currentHeartRate = ref(null)
const detectionTime = ref(null)
const detectionDuration = ref(0)
const averageHeartRate = ref(null)
const heartRateChart = ref(null)
const chartInstance = ref(null)
const heartRateData = ref([])
const recentHistory = ref([])
const username = ref('')

// 定时器引用
const detectionTimer = ref(null)
const durationTimer = ref(null)
const frameInterval = ref(null)

// 检查用户是否已登录
const checkAuth = async () => {
    const token = localStorage.getItem('access_token')
    if (!token) {
        router.push('/')
        return false
    }

    // 获取用户信息
    try {
        const response = await axios.get('/user/info')
        if (response.data && response.data.success) {
            username.value = response.data.username
        }
    } catch (error) {
        console.error('获取用户信息失败:', error)
    }

    return true
}

// 初始化摄像头
const initCamera = async () => {
    if (!checkAuth()) return

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        })

        if (videoRef.value) {
            videoRef.value.srcObject = stream
            isCameraActive.value = true
        }
    } catch (err) {
        console.error('无法访问摄像头:', err)
        isCameraActive.value = false
    }
}

// 开始检测
const startDetection = async () => {
    if (!isCameraActive.value) return

    try {
        // 增加超时时间到10秒
        const response = await axios.post('/realtime_heart_rate/start', {}, { timeout: 10000 })
        if (!response.data.success) {
            throw new Error(response.data.message)
        }

        isDetecting.value = true
        detectionTime.value = new Date()
        detectionDuration.value = 0
        heartRateData.value = []
        currentHeartRate.value = null
        averageHeartRate.value = null

        // 开始计时
        if (durationTimer.value) clearInterval(durationTimer.value)
        durationTimer.value = setInterval(() => {
            detectionDuration.value++
        }, 1000)

        // 开始发送视频帧到后端
        startFrameStreaming()

        // 初始化图表
        await nextTick()
        initChart()

        // 加载历史记录
        loadRecentHistory()
    } catch (error) {
        console.error('开始检测失败:', error)
        isDetecting.value = false
        ElMessage.error('开始检测失败: ' + (error.response?.data?.message || error.message))
    }
}

// 发送视频帧到后端
const startFrameStreaming = () => {
    if (frameInterval.value) clearInterval(frameInterval.value)

    frameInterval.value = setInterval(async () => {
        if (!isDetecting.value || !videoRef.value) return

        try {
            // 截取当前视频帧
            const canvas = document.createElement('canvas')
            const video = videoRef.value
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            const ctx = canvas.getContext('2d')
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

            // 将帧转换为base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8)

            // 发送到后端，设置更长的超时时间
            await axios.post('/realtime_heart_rate/frame', {
                frame: imageData
            }, {
                timeout: 50000 // 设置50秒超时
            })
        } catch (error) {
            console.error('发送帧数据失败:', error)
            // 添加更详细的错误信息
            if (error.code === 'ECONNABORTED') {
                console.error('请求超时，请检查网络连接或后端服务')
            } else {
                console.error('发送帧数据时发生错误:', error.message)
            }
        }
    }, 300) // 每100ms发送一帧
}

// 停止检测
const stopDetection = async () => {
    isDetecting.value = false
    isProcessing.value = false

    if (detectionTimer.value) {
        clearInterval(detectionTimer.value)
        detectionTimer.value = null
    }

    if (durationTimer.value) {
        clearInterval(durationTimer.value)
        durationTimer.value = null
    }

    if (frameInterval.value) {
        clearInterval(frameInterval.value)
        frameInterval.value = null
    }

    try {
        // 调用后端API停止检测并获取结果
        const response = await axios.post('/realtime_heart_rate/stop')
        if (response.data.success) {
            currentHeartRate.value = response.data.data.bmp
            // 保存检测记录到后端
            saveDetectionRecord(response.data.data)
        } else {
            throw new Error(response.data.message)
        }
    } catch (error) {
        console.error('停止检测失败:', error)
        ElMessage.error('停止检测失败: ' + (error.response?.data?.message || error.message))
    }
}

// 保存检测记录
const saveDetectionRecord = async (data) => {
    if (!detectionTime.value || !data.bmp) return

    try {
        // 重新加载历史记录
        loadRecentHistory()
        ElMessage.success('检测完成，结果已保存')
    } catch (error) {
        console.error('更新历史记录失败:', error)
    }
}

// 截图功能
const captureImage = () => {
    if (!videoRef.value || !canvasRef.value) return

    isProcessing.value = true
    const canvas = canvasRef.value
    const video = videoRef.value
    const ctx = canvas.getContext('2d')

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    // 模拟处理时间
    setTimeout(() => {
        isProcessing.value = false
    }, 1000)
}

// 初始化图表
const initChart = () => {
    if (!heartRateChart.value) return

    chartInstance.value = echarts.init(heartRateChart.value)
    updateChart()

    window.addEventListener('resize', handleResize)
}

// 更新图表
const updateChart = () => {
    if (!chartInstance.value || heartRateData.value.length === 0) return

    const option = {
        tooltip: {
            trigger: 'axis'
        },
        xAxis: {
            type: 'time',
            axisLabel: {
                formatter: (value) => {
                    return dayjs(value).format('HH:mm:ss')
                }
            }
        },
        yAxis: {
            type: 'value',
            min: 50,
            max: 110,
            axisLabel: {
                formatter: '{value} BPM'
            }
        },
        series: [{
            type: 'line',
            data: heartRateData.value.map(item => [item.time, item.value]),
            smooth: true,
            lineStyle: {
                width: 2,
                color: '#409EFF'
            },
            symbol: 'none',
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(64, 158, 255, 0.4)' },
                    { offset: 1, color: 'rgba(64, 158, 255, 0.05)' }
                ])
            }
        }],
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        }
    }

    chartInstance.value.setOption(option)
}

// 加载最近历史记录
const loadRecentHistory = async () => {
    try {
        const response = await axios.get('/user/health_records', {
            params: {
                page: 1,
                pageSize: 5
            }
        })

        if (response.data.success) {
            recentHistory.value = response.data.data
        }
    } catch (error) {
        console.error('加载历史记录失败:', error)
    }
}

// 获取状态类型
const getStatusType = () => {
    if (!currentHeartRate.value) return ''

    if (currentHeartRate.value < 60) return 'warning'
    if (currentHeartRate.value > 100) return 'danger'
    return 'success'
}

// 获取状态文本
const getStatusText = () => {
    if (!currentHeartRate.value) return '未开始'

    if (currentHeartRate.value < 60) return '偏低'
    if (currentHeartRate.value > 100) return '偏高'
    return '正常'
}

// 获取历史记录状态类型
const getHistoryStatusType = (bpm) => {
    if (bpm < 60) return 'warning'
    if (bpm > 100) return 'danger'
    return 'success'
}

// 获取历史记录状态文本
const getHistoryStatusText = (bpm) => {
    if (bpm < 60) return '偏低'
    if (bpm > 100) return '偏高'
    return '正常'
}

// 格式化日期时间
const formatDateTime = (date) => {
    return date ? dayjs(date).format('YYYY-MM-DD HH:mm:ss') : ''
}

// 表格中格式化日期时间
const formatDateTimeColumn = (row, column, cellValue) => {
    return dayjs(cellValue).format('YYYY-MM-DD HH:mm:ss')
}

// 格式化时长
const formatDuration = (seconds) => {
    if (!seconds) return '0秒'

    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60

    if (mins > 0) {
        return `${mins}分${secs}秒`
    }
    return `${secs}秒`
}

// 表格中格式化时长
const formatDurationColumn = (row, column, cellValue) => {
    return formatDuration(cellValue)
}

// 处理窗口大小变化
const handleResize = () => {
    if (chartInstance.value) {
        chartInstance.value.resize()
    }
}

// 组件挂载时初始化
onMounted(async () => {
    await initCamera()
    loadRecentHistory()
})

// 组件卸载前清理
onBeforeUnmount(() => {
    // 停止检测
    stopDetection()

    // 关闭摄像头
    if (videoRef.value && videoRef.value.srcObject) {
        const stream = videoRef.value.srcObject
        const tracks = stream.getTracks()
        tracks.forEach(track => track.stop())
    }

    // 移除事件监听
    window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.container {
    display: flex;
    width: 100%;
    height: 98vh;
}

.sidebar {
    width: 240px;
    height: 98vh;
    box-shadow: 2px 0 6px rgba(0, 21, 41, .35);
}

.logo {
    height: 60px;
    line-height: 60px;
    text-align: center;
    color: #fff;
    font-size: 18px;
    background-color: #2b2f3a;
}

.main-content {
    flex: 1;
    padding: 20px;
}

/* 原有样式保持不变，在下方添加： */
.el-menu-item {
    height: 56px;
    line-height: 56px;
    font-size: 14px;
}

.el-menu-item.is-active {
    background-color: #263445 !important;
}

.el-icon {
    vertical-align: middle;
    margin-right: 12px;
}

/* 新增样式 */
.camera-card, .info-card, .history-card {
    height: 100%;
}

.camera-container {
    position: relative;
    width: 100%;
    height: 350px;
    background-color: #000;
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.camera-feed, .camera-canvas {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.camera-placeholder {
    color: #ccc;
    text-align: center;
}

.control-panel {
    margin-top: 20px;
    text-align: center;
}

.heart-rate-display {
    text-align: center;
    padding: 20px 0;
    border-bottom: 1px solid #eee;
}

.heart-rate-value {
    margin-bottom: 10px;
}

.rate {
    font-size: 48px;
    font-weight: bold;
    color: #409EFF;
}

.unit {
    font-size: 18px;
    color: #909399;
    margin-left: 5px;
}

.heart-rate-chart {
    height: 200px;
    margin: 20px 0;
}

.chart-container {
    width: 100%;
    height: 100%;
}

.detection-info {
    margin-top: 20px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.history-card {
    margin-top: 20px;
}

pre {
    font-family: inherit;
    white-space: pre-wrap;
}

.el-descriptions {
    margin-bottom: 20px;
}

.el-divider {
    margin: 20px 0;
}

.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 60px;
    background: #fff;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.system-title {
    font-size: 18px;
    font-weight: 600;
    color: #304156;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 15px;
}

.user-name {
    font-size: 16px;
    color: #606266;
}
</style>
