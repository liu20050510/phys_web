<template>
  <div class="user-profile">
    <div class="user-header">
      <h2>个人资料</h2>
    </div>

    <div class="user-content">
      <div class="profile-section">
        <div class="avatar-section">
          <img :src="user.avatar || defaultAvatar" alt="用户头像" class="user-avatar">
          <button class="change-avatar-btn">更换头像</button>
        </div>

        <div class="info-section">
          <div class="info-item">
            <label>用户名:</label>
            <span>{{ user.username }}</span>
          </div>

          <div class="info-item">
            <label>邮箱:</label>
            <span>{{ user.email }}</span>
          </div>

          <div class="info-item">
            <label>注册时间:</label>
            <span>{{ formatDate(user.registerDate) }}</span>
          </div>

          <div class="info-item">
            <label>最近登录:</label>
            <span>{{ formatDate(user.lastLogin) }}</span>
          </div>
        </div>
      </div>

      <div class="history-section">
        <h3>历史记录</h3>
        <div class="history-list">
          <div
            v-for="record in historyRecords"
            :key="record.id"
            class="history-item"
          >
            <div class="history-info">
              <h4>{{ record.title }}</h4>
              <p>{{ record.description }}</p>
            </div>
            <div class="history-date">
              {{ formatDate(record.date) }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'UserProfile',
  data() {
    return {
      user: {
        username: '用户名',
        email: 'user@example.com',
        avatar: '',
        registerDate: new Date('2023-01-15'),
        lastLogin: new Date('2024-01-20 14:30:00')
      },
      historyRecords: [
        {
          id: 1,
          title: '心率检测记录1',
          description: '使用DeepPhys算法进行心率检测',
          date: new Date('2024-01-18 10:30:00')
        },
        {
          id: 2,
          title: '心率检测记录2',
          description: '使用PhysNet算法进行心率检测',
          date: new Date('2024-01-15 16:45:00')
        },
        {
          id: 3,
          title: '心率检测记录3',
          description: '使用RhythmFormer算法进行心率检测',
          date: new Date('2024-01-10 09:15:00')
        }
      ]
    }
  },
  computed: {
    defaultAvatar() {
      // 返回一个内联的SVG作为默认头像
      return 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjAiIGhlaWdodD0iMTIwIiB2aWV3Qm94PSIwIDAgMTIwIDEyMCI+PGNpcmNsZSBjeD0iNjAiIGN5PSI2MCIgcj0iNjAiIGZpbGw9IiNkZGQiLz48cGF0aCBkPSJNNjAgNjBDNDYuNiA2MCAzNiA0OS40IDM2IDM2czEwLjYtMjQgMjQtMjQgMjQgMTAuNiAyNCAyNC0xMC42IDI0LTI0IDI0eiIgZmlsbD0iIzk5OSIvPjxwYXRoIGQ9Ik05MCA5MGMtMTMuNCAwLTI0LTkuOC0yNC0yMnMyLjItMjIgMjQtMjIgMjIgOS44IDIyIDIyLTIuMiAxMi0yMiAyMnoiIGZpbGw9IiM5OTkiLz48L3N2Zz4='
    }
  },
  methods: {
    formatDate(date) {
      if (!date) return ''
      const d = new Date(date)
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
    }
  }
}
</script>

<style scoped>
.user-profile {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.user-header {
  margin-bottom: 30px;
  padding-bottom: 15px;
  border-bottom: 1px solid #e4e7ed;
}

.user-header h2 {
  color: #303133;
  font-size: 24px;
  margin: 0;
}

.user-content {
  max-width: 800px;
  margin: 0 auto;
}

.profile-section {
  display: flex;
  background: white;
  border-radius: 8px;
  padding: 30px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-bottom: 30px;
}

.avatar-section {
  flex: 0 0 150px;
  text-align: center;
  margin-right: 40px;
}

.user-avatar {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid #ebeef5;
  margin-bottom: 15px;
}

.change-avatar-btn {
  background-color: #409eff;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.change-avatar-btn:hover {
  background-color: #66b1ff;
}

.info-section {
  flex: 1;
}

.info-item {
  display: flex;
  margin-bottom: 20px;
  align-items: center;
}

.info-item label {
  width: 100px;
  font-weight: bold;
  color: #606266;
}

.info-item span {
  flex: 1;
  color: #303133;
}

.history-section {
  background: white;
  border-radius: 8px;
  padding: 30px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.history-section h3 {
  color: #303133;
  margin-top: 0;
  margin-bottom: 20px;
}

.history-list {
  max-height: 400px;
  overflow-y: auto;
}

.history-item {
  display: flex;
  justify-content: space-between;
  padding: 15px 0;
  border-bottom: 1px solid #ebeef5;
}

.history-item:last-child {
  border-bottom: none;
}

.history-info h4 {
  margin: 0 0 5px 0;
  color: #303133;
}

.history-info p {
  margin: 0;
  color: #909399;
  font-size: 14px;
}

.history-date {
  color: #c0c4cc;
  font-size: 13px;
  align-self: center;
}

@media (max-width: 768px) {
  .profile-section {
    flex-direction: column;
    text-align: center;
  }

  .avatar-section {
    margin-right: 0;
    margin-bottom: 20px;
  }

  .info-item {
    flex-direction: column;
    align-items: flex-start;
  }

  .info-item label {
    width: auto;
    margin-bottom: 5px;
  }
}
</style>
