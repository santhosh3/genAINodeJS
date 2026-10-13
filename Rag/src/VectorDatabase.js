import { PGVectorStore } from "@langchain/pgvector";

class VectorDatabase {
    constructor(pool, embeddings) {
        this.pool = pool;
        this.embeddings = embeddings;
    }

    async getStore() {
        return await PGVectorStore.initialize(this.embeddings, {
            pool: this.pool,
            tableName: "company_documents",
        });
    }

    async indexDocuments(documents) {
        const store = await this.getStore();
        await store.addDocuments(documents);
        return documents.length;
    }
}

export default VectorDatabase;