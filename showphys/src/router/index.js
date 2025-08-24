import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Auth',
    component: () => import('@/views/AuthPage.vue')
  },
  {
    path: '/Home',
    name: 'Home',
    component: () => import('@/views/HomePage.vue')
  },
  {
    path: '/History',
    name: 'History',
    component: () => import('@/views/HistoryPage.vue')
  },
  {
    path: '/About',
    name: 'About',
    component: () => import('@/views/AboutPage.vue')
  },
  {
    path: '/Welcome',
    name: 'Welcome',
    component: () => import('@/views/WelcomePage.vue')
  },
  {
    path: '/Profile',
    name: 'Profile',
    component: () => import('@/views/UserProfile.vue')
  }
  // 可以添加其他路由...
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
