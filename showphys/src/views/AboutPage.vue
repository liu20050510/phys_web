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
            <!-- 关于我们主体内容 -->
            <el-card class="about-card">
                <div class="about-content">
                    <!-- 系统介绍 -->
                    <el-row :gutter="20" class="section">
                        <el-col :span="24">
                            <h2 class="section-title">系统简介</h2>
                            <div class="section-content">
                                <p>基于多路径协同交互的远程生理监测系统是基于人工智能和生物医学工程技术开发的智能健康监测平台。系统通过非侵入式检测技术，实时采集和分析用户的生命体征数据，生成心率等关键健康指标，帮助用户了解自身健康情况。
                                </p>
                            </div>
                        </el-col>
                    </el-row>

                    <!-- 核心功能 -->
                    <el-row :gutter="20" class="section">
                        <el-col :span="24">
                            <h2 class="section-title">核心功能</h2>
                            <el-row :gutter="20">
                                <el-col :span="8">
                                    <el-card shadow="hover" class="feature-card">
                                        <div class="feature-icon"><el-icon>
                                                <Monitor />
                                            </el-icon></div>
                                        <h3>实时监测</h3>
                                        <p>支持摄像头和视频文件两种输入方式，迅速分析心率</p>
                                    </el-card>
                                </el-col>
                                <el-col :span="8">
                                    <el-card shadow="hover" class="feature-card">
                                        <div class="feature-icon"><el-icon>
                                                <DataAnalysis />
                                            </el-icon></div>
                                        <h3>智能分析</h3>
                                        <p>基于深度学习算法，自动生成健康评估报告</p>
                                    </el-card>
                                </el-col>
                                <el-col :span="8">
                                    <el-card shadow="hover" class="feature-card">
                                        <div class="feature-icon"><el-icon>
                                                <Histogram />
                                            </el-icon></div>
                                        <h3>历史追溯</h3>
                                        <p>完整记录历史数据，支持多维度趋势分析</p>
                                    </el-card>
                                </el-col>
                            </el-row>
                        </el-col>
                    </el-row>


                    <!-- 团队信息 -->
                    <el-row :gutter="20" class="section">
                        <el-col :span="24">
                            <h2 class="section-title">研发团队</h2>
                            <el-row :gutter="20">
                                <el-col :span="8" v-for="member in teamMembers" :key="member.name">
                                    <el-card class="member-card">
                                        <div class="avatar">
                                            <el-avatar :size="80" :src="member.avatar" />
                                        </div>
                                        <h3>{{ member.name }}</h3>
                                        <p class="position">{{ member.position }}</p>
                                        <p class="desc">{{ member.description }}</p>
                                    </el-card>
                                </el-col>
                            </el-row>
                        </el-col>
                    </el-row>
                </div>
            </el-card>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
    House,
    Clock,
    InfoFilled,
    Monitor,
    DataAnalysis,
    Histogram,
    User,
    Odometer
} from '@element-plus/icons-vue'
import axios from '@/apis/axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const activeMenu = ref('/About')
const username = ref('') // 添加用户名变量

const teamMembers = ref([
    {
        name: '肖昌昊',
        position: '团队负责人',
        avatar: 'https://example.com/avatar1.jpg',
        description: '合肥工业大学软件学院23级本科生，负责团队统筹与算法模型优化'
    },
    {
        name: '柳虹宇',
        position: '软件开发',
        avatar: 'https://example.com/avatar2.jpg',
        description: '合肥工业大学软件学院23级本科生，负责系统架构设计与全栈开发'
    },
    {
        name: '年付盛',
        position: '文书撰写',
        avatar: 'https://example.com/avatar3.jpg',
        description: '合肥工业大学软件学院23级本科生，负责各项项目材料的总结与撰写'
    }
])

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

onMounted(() => {
    fetchUserInfo() // 获取用户信息
})
</script>

<style scoped>
.container {
    display: flex;
    height: 98vh;
    width: 100%;
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

.about-card {
    min-height: calc(100vh - 160px);
}

.section {
    margin-bottom: 0px;
    padding: 5px 0;
}

.section-title {
    color: #409EFF;
    border-bottom: 2px solid #409EFF;
    padding-bottom: 5px;
    margin-bottom: 10px;
}

.feature-card {
    text-align: center;
    height: 180px;
    transition: transform 0.3s;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-icon {
    font-size: 40px;
    color: #409EFF;
    margin: 15px 0;
}

.architecture {
    display: flex;
    gap: 30px;
    align-items: center;
}

.architecture img {
    width: 50%;
    border-radius: 4px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.architecture-desc {
    flex: 1;
    line-height: 1.8;
}

.member-card {
    text-align: center;
    padding: 10px;
}

.member-card .avatar {
    margin-bottom: 15px;
}

.position {
    color: #666;
    margin: 10px 0;
}

.desc {
    font-size: 0.9em;
    color: #999;
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
