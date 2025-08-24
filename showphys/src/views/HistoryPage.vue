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
            <!-- 搜索过滤栏 -->
            <el-card class="filter-bar">
                <el-form :inline="true">
                    <el-form-item label="日期范围">
                        <el-date-picker v-model="dateRange" type="daterange" range-separator="至"
                            start-placeholder="开始日期" end-placeholder="结束日期" value-format="YYYY-MM-DD" />
                    </el-form-item>
                    <el-form-item>
                        <el-button type="primary" @click="loadData">查询</el-button>
                    </el-form-item>
                </el-form>
            </el-card>

            <!-- 主要数据区域 -->
            <el-row :gutter="20">
                <!-- 历史记录表格 -->
                <el-col :span="14">
                    <el-card>
                        <el-table :data="historyData" stripe style="width: 100%" height="calc(100vh - 330px)">
                            <el-table-column prop="detect_time" label="检测时间" width="250" sortable
                                :formatter="formatDateTime" />
                            <el-table-column prop="bmp" label="心率(BPM)" width="200" />
                            <el-table-column label="操作" width="200">
                                <template #default="scope">
                                    <el-button link type="primary" @click="showDetail(scope.row)">查看详情</el-button>
                                </template>
                            </el-table-column>
                        </el-table>
                        <el-pagination class="pagination" v-model:current-page="currentPage" :page-size="pageSize"
                            :total="total" layout="total, prev, pager, next" @current-change="handlePageChange" />
                    </el-card>
                </el-col>

                <!-- 趋势图表 -->
                <el-col :span="10">
                    <el-card class="chart-card">
                        <div ref="trendChart" class="chart-container"></div>
                        <!-- 添加统计信息 -->
                        <div class="stats-container">
                            <el-row :gutter="10">
                                <el-col :span="8">
                                    <div class="stat-item">
                                        <div class="stat-label">平均心率</div>
                                        <div class="stat-value">{{ avgBpm }} BPM</div>
                                    </div>
                                </el-col>
                                <el-col :span="8">
                                    <div class="stat-item">
                                        <div class="stat-label">最高心率</div>
                                        <div class="stat-value high">{{ maxBpm }} BPM</div>
                                    </div>
                                </el-col>
                                <el-col :span="8">
                                    <div class="stat-item">
                                        <div class="stat-label">最低心率</div>
                                        <div class="stat-value low">{{ minBpm }} BPM</div>
                                    </div>
                                </el-col>
                            </el-row>
                        </div>
                    </el-card>
                </el-col>
            </el-row>
        </div>

        <!-- 详情对话框 -->
        <el-dialog v-model="detailVisible" title="检测详情" width="60%">
            <el-descriptions :column="2" border>
                <el-descriptions-item label="检测时间" label-align="right" align="center">
                    <el-tag type="success">{{ currentDetail.detect_time }}</el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="心率值" label-align="right" align="center">
                    <el-tag type="danger">{{ currentDetail.bmp }} BPM</el-tag>
                </el-descriptions-item>
            </el-descriptions>
            <el-divider />
            <h4 style="color: #409EFF; margin-bottom: 15px;">详细分析报告：</h4>
            <el-scrollbar height="300px">
                <pre style="font-family: 'Consolas', monospace; line-height: 1.6;">{{ currentDetail.report }}</pre>
            </el-scrollbar>

            <template #footer>
                <el-button type="primary" @click="detailVisible = false">关闭</el-button>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref, onMounted, watch, computed } from 'vue'
import * as echarts from 'echarts'
import dayjs from 'dayjs'
import {
    House,
    Clock,
    InfoFilled,
    User,
    Odometer
} from '@element-plus/icons-vue'
import { nextTick } from 'vue'
import axios from '@/apis/axios'
import { useRouter } from 'vue-router'

const router = useRouter()

// 数据与状态
const activeMenu = ref('/History')
const dateRange = ref([])
const historyData = ref([])
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const chartInstance = ref(null)
const detailVisible = ref(false)
const currentDetail = ref({})
const trendChart = ref(null) // 声明图表容器引用
const username = ref('') // 添加用户名变量

// 计算统计值
const avgBpm = computed(() => {
    if (historyData.value.length === 0) return 0
    const sum = historyData.value.reduce((acc, item) => acc + item.bmp, 0)
    return Math.round(sum / historyData.value.length)
})

const maxBpm = computed(() => {
    if (historyData.value.length === 0) return 0
    return Math.max(...historyData.value.map(item => item.bmp))
})

const minBpm = computed(() => {
    if (historyData.value.length === 0) return 0
    return Math.min(...historyData.value.map(item => item.bmp))
})

// 获取用户信息
const fetchUserInfo = async () => {
    try {
        const token = localStorage.getItem('access_token')
        if (!token) {
            router.push('/')
            return
        }

        const response = await axios.get('/user/info')
        if (response.data && response.data.success) {
            username.value = response.data.username
        }
    } catch (error) {
        console.error('获取用户信息失败:', error)
    }
}

// 加载数据
const loadData = async () => {
    try {
        // 构建请求参数
        const params = {
            page: currentPage.value,
            pageSize: pageSize.value
        }

        // 添加日期范围参数
        if (dateRange.value && dateRange.value.length === 2) {
            params.startDate = dateRange.value[0]
            params.endDate = dateRange.value[1]
        }

        // 从后端获取数据
        const response = await axios.get('/user/health_records', { params })

        if (response.data.success) {
            historyData.value = response.data.data
            total.value = response.data.total
        } else {
            console.error('获取数据失败:', response.data.message)
        }

        // 初始化图表
        await nextTick()
        if (!chartInstance.value && trendChart.value) {
            chartInstance.value = echarts.init(trendChart.value)
            window.addEventListener('resize', () => chartInstance.value.resize())
        }
        updateChart(historyData.value)

    } catch (error) {
        console.error('加载数据失败:', error)
    }
}

// 处理分页变化
const handlePageChange = (page) => {
    currentPage.value = page
    loadData()
}

// 更新图表函数
// 增强图表配置
const updateChart = (data) => {
    // 准备数据
    const chartData = data.map(item => [dayjs(item.detect_time).valueOf(), item.bmp])

    // 计算平均值
    const avgValue = chartData.length > 0 ?
        chartData.reduce((sum, item) => sum + item[1], 0) / chartData.length : 0

    const option = {
        tooltip: {
            trigger: 'axis',
            formatter: (params) => {
                const date = dayjs(params[0].value[0]).format('YYYY-MM-DD HH:mm')
                const value = params[0].value[1]
                return `${date}<br/>心率: ${value} BPM`
            }
        },
        legend: {
            data: ['心率', '平均值'],
            top: 10
        },
        xAxis: {
            type: 'time',
            axisLabel: {
                formatter: (value) => {
                    const date = dayjs(value)
                    // 动态格式调整
                    if (date.hour() === 0 && date.minute() === 0) {
                        return date.format('MM/DD')
                    }
                    return date.format('HH:mm')
                },
                rotate: 45, // 标签旋转45度
                margin: 15, // 增加标签间距
                interval: (index, value) => {
                    // 智能间隔控制
                    const range = dayjs().diff(dayjs(value), 'hour')
                    return range > 72 ? 24 : range > 24 ? 6 : 2
                }
            },
            axisTick: {
                alignWithLabel: true // 刻度与标签对齐
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
        series: [
            {
                name: '心率',
                type: 'line',
                data: chartData,
                smooth: true,
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(64, 158, 255, 0.4)' },
                        { offset: 1, color: 'rgba(64, 158, 255, 0.05)' }
                    ])
                },
                lineStyle: {
                    width: 3,
                    color: '#409EFF'
                },
                symbol: 'circle',
                symbolSize: 8,
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '平均值',
                type: 'line',
                data: chartData.map(item => [item[0], avgValue]),
                markLine: {
                    silent: true,
                    lineStyle: {
                        type: 'dashed',
                        color: '#E6A23C'
                    },
                    symbol: 'none',
                    label: {
                        position: 'middle',
                        formatter: '平均值: ' + avgValue.toFixed(1) + ' BPM'
                    },
                    data: [
                        {
                            yAxis: avgValue
                        }
                    ]
                }
            }
        ],
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        }
    }

    chartInstance.value.setOption(option, true)
}

const showDetail = (row) => {
    currentDetail.value = {
        ...row,
        detect_time: dayjs(row.detect_time).format('YYYY-MM-DD HH:mm:ss')
    }
    detailVisible.value = true
}

// 添加窗口resize监听
window.addEventListener('resize', () => {
    chartInstance.value?.resize()
})

watch([currentPage], () => {
    loadData()
})

watch(dateRange, () => {
    currentPage.value = 1 // 切换日期时重置到第一页
    loadData()
})

// 添加初始化代码
onMounted(async () => {
    await fetchUserInfo() // 获取用户信息
    await loadData()
    window.addEventListener('resize', () => chartInstance.value?.resize())
})

// 添加时间格式化方法
const formatDateTime = (row, column, cellValue) => {
    return dayjs(cellValue).format('YYYY-MM-DD HH:mm:ss')
}
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

/* 自定义样式 */
.filter-bar {
    margin-bottom: 20px;
}

.chart-card {
    height: calc(100vh - 240px);
}

.chart-container {
    height: 300px;
    width: 100%;
    margin-top: 10px;
}

.pagination {
    margin-top: 20px;
    justify-content: flex-end;
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

.stats-container {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid #eee;
}

.stat-item {
    text-align: center;
    padding: 10px 0;
}

.stat-label {
    font-size: 14px;
    color: #909399;
    margin-bottom: 5px;
}

.stat-value {
    font-size: 20px;
    font-weight: bold;
    color: #409EFF;
}

.stat-value.high {
    color: #F56C6C;
}

.stat-value.low {
    color: #67C23A;
}
</style>
