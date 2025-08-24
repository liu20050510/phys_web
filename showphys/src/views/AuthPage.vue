<template>
    <div class="auth-container">
        <div class="system-banner">
            <div class="banner-overlay">
                <h1 class="system-title">基于多路径协同交互的远程生理监测系统</h1>
            </div>
        </div>
        <el-card class="auth-card">
            <h2 class="title">{{ isLogin ? '用户登录' : '用户注册' }}</h2>

            <el-form :model="formData" :rules="formRules" ref="authForm" @submit.prevent="handleSubmit">
                <!-- 注册时显示用户名输入 -->
                <el-form-item v-if="!isLogin" prop="username">
                    <el-input v-model="formData.username" placeholder="请输入用户名" clearable>
                        <template #prefix>
                            <el-icon>
                                <User />
                            </el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="email">
                    <el-input v-model="formData.email" placeholder="请输入邮箱" clearable>
                        <template #prefix>
                            <el-icon>
                                <Message />
                            </el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item prop="password">
                    <el-input v-model="formData.password" type="password" placeholder="请输入密码" show-password>
                        <template #prefix>
                            <el-icon>
                                <Lock />
                            </el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <!-- 注册时显示确认密码 -->
                <el-form-item v-if="!isLogin" prop="confirmPassword">
                    <el-input v-model="formData.confirmPassword" type="password" placeholder="请确认密码" show-password>
                        <template #prefix>
                            <el-icon>
                                <Lock />
                            </el-icon>
                        </template>
                    </el-input>
                </el-form-item>

                <el-form-item>
                    <el-button class="submit-btn" type="primary" native-type="submit">
                        {{ isLogin ? '立即登录' : '立即注册' }}
                    </el-button>
                </el-form-item>
            </el-form>

            <div class="switch-text">
                {{ isLogin ? '没有账号？' : '已有账号？' }}
                <a href="javascript:;" @click="toggleForm">
                    {{ isLogin ? '立即注册' : '立即登录' }}
                </a>
            </div>
        </el-card>
    </div>
</template>

<script setup>
import { ref, reactive } from 'vue';
import { ElMessage } from 'element-plus';
import { User, Lock, Message } from '@element-plus/icons-vue';
import { useRouter } from 'vue-router';
import { register, login } from '../apis/auth'; // 引入 API

const isLogin = ref(true);
const authForm = ref(null);

const formData = reactive({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
});

const validateConfirmPassword = (rule, value, callback) => {
    if (value !== formData.password) {
        callback(new Error('两次输入的密码不一致'));
    } else {
        callback();
    }
};

const formRules = reactive({
    username: [
        { required: true, message: '请输入用户名', trigger: 'blur' },
        { min: 3, max: 12, message: '长度在 3 到 12 个字符', trigger: 'blur' }
    ],
    email: [
        { required: true, message: '请输入邮箱地址', trigger: 'blur' },
        { type: 'email', message: '请输入正确的邮箱格式', trigger: ['blur', 'change'] }
    ],
    password: [
        { required: true, message: '请输入密码', trigger: 'blur' },
        { min: 6, max: 16, message: '长度在 6 到 16 个字符', trigger: 'blur' }
    ],
    confirmPassword: [
        { required: true, message: '请再次输入密码', trigger: 'blur' },
        { validator: validateConfirmPassword, trigger: 'blur' }
    ]
});

const toggleForm = () => {
    isLogin.value = !isLogin.value;
    authForm.value.resetFields();
};

const router = useRouter();

const handleSubmit = async () => {
    try {
        await authForm.value.validate();

        if (isLogin.value) {
            const { email, password } = formData;
            const response = await login({ email, password });

            // 调试输出完整响应
            console.log('[DEBUG] 登录响应:', response);

            // 验证响应结构
            if (!response?.access_token) {
                throw new Error('服务器未返回有效Token');
            }

            // 存储Token并验证
            localStorage.setItem('access_token', response.access_token);
            console.log('[DEBUG] 存储验证:', localStorage.getItem('access_token'));

            // 跳转前延迟确保存储完成
            await new Promise(resolve => setTimeout(resolve, 50));

            ElMessage.success('登录成功，正在跳转...');
            router.push('/Welcome');
        } else {
            // 注册逻辑保持不变
        }
    } catch (error) {
        console.error('[ERROR] 登录失败详情:', {
            error: error.response?.data || error.message,
            stack: error.stack
        });

        const errorMsg = error.response?.data?.message
            || error.message
            || '未知错误';

        ElMessage.error(`操作失败: ${errorMsg}`);
    }
};
</script>

<style scoped>
.auth-card {
    width: 400px;
    padding: 20px;
}

.title {
    font-size: 1.8rem;
    color: #304156;
    margin-bottom: 30px;
}

.submit-btn {
    width: 100%;
    margin-top: 10px;
}

.switch-text {
    text-align: right;
    margin-top: 15px;
    color: #666;
}

.switch-text a {
    color: #409eff;
    text-decoration: none;
}

.switch-text a:hover {
    text-decoration: underline;
}

.el-form-item {
    margin-bottom: 22px;
}

.auth-container {
    position: relative;
    height: 98vh;
    background:
        linear-gradient(rgba(0, 0, 0, 0.0), rgba(0, 0, 0, 0.0)),
        url('/src/images/登录页背景.png') center/cover fixed;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100vw;
    overflow: hidden;
}

/* 背景图片优化 */
.auth-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: inherit;
    z-index: 0;
    transform: scale(1.02);
    /* 消除边缘白边 */
}

.system-banner {
    position: absolute;
    top: 10%;
    left: 0;
    right: 0;
    z-index: 1;
}

.banner-overlay {
    background: rgba(0, 0, 0, 0.4);
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.system-title {
    color: #fff;
    font-size: 2.2rem;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    max-width: 1200px;
    line-height: 1.3;
    padding: 0 20px;
}

.auth-card {
    width: 400px;
    /* 保持合适宽度 */
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    background: rgba(255, 255, 255, 0.96);
    position: relative;
    z-index: 2;
    margin: 20px;
    /* 增加边距防止触屏 */
}

/* 响应式调整 */
@media (max-width: 768px) {
    .system-title {
        font-size: 1.8rem;
        padding: 0 15px;
    }

    .auth-card {
        width: 90%;
        max-width: 400px;
        transform: translateY(60px);
        /* 下移避免标题遮挡 */
    }
}

@media (max-width: 480px) {
    .system-title {
        font-size: 1.4rem;
        line-height: 1.4;
    }

    .auth-card {
        padding: 20px;
    }

    .title {
        font-size: 1.6rem;
    }
}
</style>