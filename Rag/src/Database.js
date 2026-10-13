import { Pool } from "pg";


class Database {
  constructor() {
    if (!Database.instance) {
      this.pool = new Pool({
        connectionString: "postgresql://admin:admin@localhost:5432/ragdb",
        max: 20, 
        idleTimeoutMillis: 30000,
        connectionTimeoutMillis: 2000,
      });

      Database.instance = this;
    }
    return Database.instance;
  }

  getPool() {
    return this.pool;
  }
}

export default new Database();