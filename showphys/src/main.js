import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'

const app = createApp(App)

// 注册Element Plus
app.use(ElementPlus)

//使用路由
app.use(router)

app.mount('#app')