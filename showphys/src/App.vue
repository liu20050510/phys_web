<template>
  <router-view v-if="isRouterAlive" />
</template>

<script setup>
import { ref } from 'vue' // 确保导入 nextTick

const isRouterAlive = ref(true)
</script>

<style>
/* 全局样式 */
#app {
  height: 100%;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #f0f2f5;
}
</style>
