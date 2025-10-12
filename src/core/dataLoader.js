class DataLoader {
    loadCSV(filePath, options = {}) {
        const defaultOptions = {
            delimiter: ',',
            header: true,
            skipEmptyLines: true,
            encoding: 'utf8'
        };

        const config = { ...defaultOptions, ...options };

        try {
            if (typeof window !== 'undefined' && window.fs) {
                const content = window.fs.readFileSync ?
                    window.fs.readFileSync(filePath, { encoding: config.encoding }) :
                    window.fs.readFile(filePath, { encoding: config.encoding });

                return this.parseCSV(content, config);
            } else {
                throw new Error('File system not available');
            }
        } catch (error) {
            throw new Error(`Failed to load CSV: ${error.message}`);
        }
    }

    loadJSON(jsonInput, options = {}) {
        const defaultOptions = {
            validateTypes: true,
            autoInferHeaders: true
        };

        const config = { ...defaultOptions, ...options };

        try {
            let jsonData;

            if (typeof jsonInput === 'string') {
                if (jsonInput.endsWith('.json') && typeof window !== 'undefined' && window.fs) {
                    const content = window.fs.readFileSync ?
                        window.fs.readFileSync(jsonInput, { encoding: 'utf8' }) :
                        window.fs.readFile(jsonInput, { encoding: 'utf8' });
                    jsonData = JSON.parse(content);
                } else {
                    jsonData = JSON.parse(jsonInput);
                }
            } else if (typeof jsonInput === 'object') {
                jsonData = jsonInput;
            } else {
                throw new Error('Invalid JSON input: must be string, file path, or object');
            }

            return this.parseJSON(jsonData, config);
        } catch (error) {
            throw new Error(`Failed to load JSON: ${error.message}`);
        }
    }

    parseJSON(jsonData, config) {
        if (!jsonData) {
            throw new Error('JSON data is empty or null');
        }

        if (Array.isArray(jsonData)) {
            return this.parseJSONArray(jsonData, config);
        } else if (jsonData.headers && jsonData.data) {
            return this.parseStructuredJSON(jsonData, config);
        } else if (typeof jsonData === 'object') {
            return this.parseJSONObject(jsonData, config);
        } else {
            throw new Error('Unsupported JSON format');
        }
    }

    parseJSONArray(jsonArray, config) {
        if (jsonArray.length === 0) {
            throw new Error('JSON array is empty');
        }

        const firstRow = jsonArray[0];
        if (typeof firstRow !== 'object' || firstRow === null) {
            throw new Error('JSON array must contain objects');
        }

        let headers;
        if (config.autoInferHeaders) {
            const allKeys = new Set();
            jsonArray.forEach(row => {
                if (typeof row === 'object' && row !== null) {
                    Object.keys(row).forEach(key => allKeys.add(key));
                }
            });
            headers = Array.from(allKeys);
        } else {
            headers = Object.keys(firstRow);
        }

        const data = jsonArray.map((row, index) => {
            if (typeof row !== 'object' || row === null) {
                console.warn(`Row ${index} is not an object, skipping`);
                return null;
            }

            const processedRow = {};
            headers.forEach(header => {
                let value = row[header];
                if (config.validateTypes) {
                    value = this.inferType(value);
                }
                processedRow[header] = value;
            });
            return processedRow;
        }).filter(row => row !== null);

        return {
            headers,
            data,
            length: data.length,
            columns: headers.length,
            source: 'json_array'
        };
    }

    parseStructuredJSON(jsonData, config) {
        const { headers, data } = jsonData;

        if (!Array.isArray(headers)) {
            throw new Error('Headers must be an array');
        }

        if (!Array.isArray(data)) {
            throw new Error('Data must be an array');
        }

        if (headers.length === 0) {
            throw new Error('Headers array is empty');
        }

        const processedData = data.map((row, index) => {
            if (Array.isArray(row)) {
                const processedRow = {};
                headers.forEach((header, i) => {
                    let value = i < row.length ? row[i] : null;
                    if (config.validateTypes) {
                        value = this.inferType(value);
                    }
                    processedRow[header] = value;
                });
                return processedRow;
            } else if (typeof row === 'object' && row !== null) {
                const processedRow = {};
                headers.forEach(header => {
                    let value = row[header];
                    if (config.validateTypes) {
                        value = this.inferType(value);
                    }
                    processedRow[header] = value;
                });
                return processedRow;
            } else {
                console.warn(`Row ${index} has invalid format, skipping`);
                return null;
            }
        }).filter(row => row !== null);

        return {
            headers,
            data: processedData,
            length: processedData.length,
            columns: headers.length,
            source: 'structured_json'
        };
    }

    parseJSONObject(jsonObject, config) {
        const entries = Object.entries(jsonObject);
        if (entries.length === 0) {
            throw new Error('JSON object is empty');
        }

        const headers = ['key', 'value'];
        const data = entries.map(([key, value]) => ({
            key: key,
            value: config.validateTypes ? this.inferType(value) : value
        }));

        return {
            headers,
            data,
            length: data.length,
            columns: 2,
            source: 'json_object'
        };
    }

    parseCSV(content, options) {
        const lines = content.split('\n').filter(line =>
            options.skipEmptyLines ? line.trim() !== '' : true
        );

        if (lines.length === 0) {
            throw new Error('CSV file is empty');
        }

        const headers = options.header ?
            lines[0].split(options.delimiter).map(h => h.trim().replace(/['"]/g, '')) :
            Array.from({ length: lines[0].split(options.delimiter).length }, (_, i) => `col_${i}`);

        const startIndex = options.header ? 1 : 0;
        const data = [];

        for (let i = startIndex; i < lines.length; i++) {
            const values = lines[i].split(options.delimiter);
            if (values.length === headers.length) {
                const row = {};
                headers.forEach((header, index) => {
                    let value = values[index].trim().replace(/['"]/g, '');
                    row[header] = this.inferType(value);
                });
                data.push(row);
            }
        }

        return {
            headers,
            data,
            length: data.length,
            columns: headers.length
        };
    }

    inferType(value) {
        if (value === '' || value === 'null' || value === 'NULL' || value === 'NaN') {
            return null;
        }

        if (value === 'true' || value === 'TRUE') return true;
        if (value === 'false' || value === 'FALSE') return false;

        if (/^-?\d+$/.test(value)) {
            return parseInt(value, 10);
        }

        if (/^-?\d*\.\d+$/.test(value)) {
            return parseFloat(value);
        }

        return value;
    }

    cleanData(dataset) {
        const cleaned = {
            ...dataset,
            data: dataset.data.filter(row => {
                return Object.values(row).some(value => value !== null && value !== undefined);
            })
        };

        cleaned.length = cleaned.data.length;
        return cleaned;
    }

    getDataInfo(dataset) {
        const info = {
            rows: dataset.length,
            columns: dataset.columns,
            headers: dataset.headers,
            types: {},
            nullCounts: {},
            uniqueCounts: {}
        };

        dataset.headers.forEach(header => {
            const column = dataset.data.map(row => row[header]);
            const nonNullValues = column.filter(val => val !== null && val !== undefined);
            const types = [...new Set(nonNullValues.map(val => typeof val))];

            info.types[header] = types.length === 1 ? types[0] : 'mixed';
            info.nullCounts[header] = column.length - nonNullValues.length;
            info.uniqueCounts[header] = new Set(nonNullValues).size;
        });

        return info;
    }

    getColumn(dataset, columnName) {
        if (!dataset.headers.includes(columnName)) {
            throw new Error(`Column '${columnName}' not found`);
        }

        return dataset.data
            .map(row => row[columnName])
            .filter(val => val !== null && val !== undefined && !isNaN(val));
    }

    getColumns(dataset, columnNames) {
        const result = {};
        columnNames.forEach(name => {
            result[name] = this.getColumn(dataset, name);
        });
        return result;
    }

    filterRows(dataset, condition) {
        return {
            ...dataset,
            data: dataset.data.filter(condition),
            length: dataset.data.filter(condition).length
        };
    }

    sortBy(dataset, columnName, order = 'asc') {
        const sortedData = [...dataset.data].sort((a, b) => {
            const aVal = a[columnName];
            const bVal = b[columnName];

            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;

            if (typeof aVal === 'string' && typeof bVal === 'string') {
                return order === 'asc' ?
                    aVal.localeCompare(bVal) :
                    bVal.localeCompare(aVal);
            }

            return order === 'asc' ? aVal - bVal : bVal - aVal;
        });

        return {
            ...dataset,
            data: sortedData
        };
    }
}

export default DataLoader;