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
            <!-- 新增背景模块 -->
            <div class="welcome-module">
                <div class="content-overlay">
                    <el-button class="start-btn" type="primary" size="large" @click="navigateToMonitor">
                        开始检测
                    </el-button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
    House,
    Clock,
    InfoFilled,
    User,
    Odometer
} from '@element-plus/icons-vue'
import { useRouter } from 'vue-router'
import axios from '@/apis/axios'

const router = useRouter()
const activeMenu = ref('/Welcome')
const username = ref('')

const navigateToMonitor = () => {
    router.push('/Home') // 确保路由配置正确
}

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
    fetchUserInfo()
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

.welcome-module {
    height: calc(100vh - 160px);
    /* 根据标题栏高度调整 */
    background:
        linear-gradient(rgba(0, 0, 0, 0.0), rgba(0, 0, 0, 0.0)),
        url('/src/images/Welcome.jpg') center/cover;
    border-radius: 12px;
    position: relative;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.content-overlay {
    position: absolute;
    width: 100%;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    padding-bottom: 0px;
    /* 控制按钮垂直位置 */
}

.start-btn {
    padding: 30px 50px;
    font-size: 2rem;
    letter-spacing: 2px;
    border-radius: 30px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(64, 158, 255, 0.4);
    position: absolute;
    bottom: 25%;
    /* 数值越大位置越靠上 */
    left: 41.4%;
}

.start-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(64, 158, 255, 0.6);
}

/* 响应式调整 */
@media (max-width: 768px) {
    .welcome-module {
        height: calc(100vh - 140px);
    }

    .start-btn {
        padding: 15px 30px;
        font-size: 1.2rem;
    }

    .content-overlay {
        padding-bottom: 60px;
    }
}

@media (max-width: 480px) {
    .start-btn {
        width: 80%;
        padding: 12px 20px;
    }
}
</style>
