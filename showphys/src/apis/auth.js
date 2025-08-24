import instance from './axios';

// 注册
export const register = async (data) => {
    try {
        const response = await instance.post('/register', data);
        return response.data;
    } catch (error) {
        console.error('注册失败:', error);
        throw error;
    }
};

// 登录
export const login = async (data) => {
    try {
        const response = await instance.post('/login', data);
        return response.data;
    } catch (error) {
        console.error('登录失败:', error);
        throw error;
    }
};