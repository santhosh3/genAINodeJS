import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import csv from "csvtojson";
import fs from "node:fs/promises";
import path from "node:path";

class DocumentProcessor {
    constructor(folderPath, jsonOutputPath = "./docs/store.json") {
        this.folderPath = folderPath;
        this.jsonOutputPath = path.resolve(jsonOutputPath);
        this.splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 600,
            chunkOverlap: 100
        });
    }

    async process() {
        await this._generateJsonStore();

        // 2. Load and split documents for pgvector embedding
        const loader = new DirectoryLoader(this.folderPath, {
            ".txt": (p) => new TextLoader(p),
            ".md": (p) => new TextLoader(p),
            ".csv": (p) => new CSVLoader(p)
        }, true);

        const docs = await loader.load();

        const enrichedDocs = docs.map(doc => ({
            ...doc,
            metadata: {
                ...doc.metadata,
                folder: path.basename(path.dirname(doc.metadata.source))
            }
        }));

        return await this.splitter.splitDocuments(enrichedDocs);
    }

    async _generateJsonStore() {
        try {
            const files = await fs.readdir(this.folderPath);
            const allJsonRows = [];

            for (const file of files) {
                if (file.endsWith(".csv")) {
                    const filePath = path.join(this.folderPath, file);
                    const jsonRows = await csv().fromFile(filePath);
                    allJsonRows.push(...jsonRows);
                }
            }

            if (allJsonRows.length > 0) {
                await fs.mkdir(path.dirname(this.jsonOutputPath), { recursive: true });
                await fs.writeFile(this.jsonOutputPath, JSON.stringify(allJsonRows, null, 2));
                console.log(`[DocumentProcessor] JSON dataset created at: ${this.jsonOutputPath}`);
            }
        } catch (error) {
            console.error("[DocumentProcessor] Error generating JSON store:", error.message);
        }
    }
}

export default DocumentProcessor;