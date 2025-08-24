import axios from 'axios';

const instance = axios.create({
    baseURL: 'http://localhost:5000', // Flask 后端的地址
    timeout: 5000,
});

// 添加请求拦截器
instance.interceptors.request.use(
    (config) => {
        // 从localStorage获取token
        const token = localStorage.getItem('access_token');
        if (token) {
            // 在请求头中添加Authorization字段
            config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 添加响应拦截器
instance.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response && error.response.status === 401) {
            // 如果是401错误，清除本地token并跳转到登录页
            localStorage.removeItem('access_token');
            window.location.href = '/#/'; // 跳转到登录页
        }
        return Promise.reject(error);
    }
);

export default instance;
