const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// request config
const REQUEST_LIMIT = 100;
const CONCURRENCY_LIMIT = 3;
const MAX_RETRIES = 5;
const RETRY_TIME = 2000;

// data dir
const DATA_DIR = path.join(__dirname, "data");

// date format YYYYMMDD according to (https://open.fda.gov/apis/food/event/example-api-queries/)
const DATE_START = "20000101";
const DATE_END = "20260101";

//API Key, good for 120,000 requests/day
const API_KEY = "";

// -------------------------------------------------------------------------------------------------- //

// split by year
/*
 * Note: for some reason api does not give data past September 2025, I don't know why.
 * The website says it is updated quarterly but there are no results with date_start or date_created 
 */
function generateYearRanges(){
    const start = parseInt(DATE_START.substring(0,4), 10);
    const end = parseInt(DATE_END.substring(0, 4), 10);

    let ranges = [];
    for(let year = start; year < end; year++){
        const from = "" + year + "0101"; // jan 01 start
        const to = year === end - 1 ? DATE_END : "" + year + "1231"; // end on either 01/26 or 12/31
        ranges.push({from, to, label: "" + year});
    }
    return ranges;
}

function buildUrl(from, to){
    const base = "https://api.fda.gov/food/event.json";
    const search = `date_started:[${from}+TO+${to}]`;
    let query = `search=${search}&limit=${REQUEST_LIMIT}&sort=date_started:asc`;
    if(API_KEY){
        query = `api_key=${API_KEY}&${query}`;
    }
    return `${base}?${query}`;
}

function httpGet(requestUrl){
    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(requestUrl);
        const lib = parsedUrl.protocol === "https:" ? https : http;

        const req = lib.get(requestUrl, (res) => {
            let data = "";
            res.on("data", (chunk) => data += chunk);
            res.on("end",  () => {
                resolve({
                    statusCode: res.statusCode,
                    headers: res.headers,
                    body: data
                });
            });
        });

        req.on("error", (err) => reject(err));
        req.setTimeout(30000, () => { // timeout if it hangs for 30 seconds
            req.destroy(new Error("Request timed out after 30 seconds"))
        });
    });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(requestUrl, retries = MAX_RETRIES){
    for(let attempt = 0; attempt < retries; attempt++){
        try{
            const response = await httpGet(requestUrl);

            if(response.statusCode >= 200 && response.statusCode < 300){
                return response;
            }

            if(response.statusCode === 404){ // not found, not an error
                return response;
            }

            if(response.statusCode === 429 || response.statusCode >= 500){
                const last = attempt === retries;
                if(last){
                    throw new Error("Status: [" + response.statusCode + "] on url [" + requestUrl + "] after [" + (attempt + 1) + "] tries");
                }
                await sleep(RETRY_TIME);
                continue;
            }

            // some other error
            throw new Error("Status: [" + response.statusCode + "] on url [" + requestUrl + "] body [\n" + (response.body.slice(0, 300)) + "\n]");
        }catch(err){
            const last = attempt === retries;
            if(last){ // give up if last attempt
                throw err;
            }
            console.warn(err.message);
            await sleep(RETRY_TIME);
        }
    }
}

function extractNextUrl(linkHeader){
    if(!linkHeader) return null;

    const parts = linkHeader.split(",");
    for(const part of parts){
        const match = part.match(/<([^>]+)>;\s*rel="next"/i);
        if(match){
            return match[1];
        }
    }
    return null;
}

async function downloadDateRange(from, to, label){
    let allRecords = [];
    let currentUrl = buildUrl(from, to);
    let pageNum = 0;
    let totalExpected = null;

    while(currentUrl){
        pageNum++;
        const response = await fetchWithRetry(currentUrl);
        
        if(response.statusCode === 404){
            console.log("Found no records for lable [" + label + "] (404)");
            break;
        }

        let json;
        try{
            json = JSON.parse(response.body);
        }catch(err){
            console.error("Failed to parse json in label [" + label + "] on page [" + pageNum + "]: " + err.message);
            break;
        }

        if(totalExpected === null && json.meta && json.meta.results){
            totalExpected = json.meta.results.total;
            console.log("[" + label + "] records to download: " + totalExpected);
        }

        if(json.results && json.results.length > 0){
            allRecords.push(...json.results);
        }

        if(pageNum % 10 === 0){
            console.log("[" + label + "] Progress: " + allRecords.length + (totalExpected ? "/" + totalExpected : "") + " records page ("+pageNum+")");
        }

        const nextUrl = extractNextUrl(response.headers["link"]);

        if(!nextUrl){
            break;
        }

        currentUrl = nextUrl;
        await(sleep(100));
    }
    console.log("[" + label + "] downloaded ["+allRecords.length+"] records in [" + pageNum + "] page(s)");
    return allRecords;
}

function saveRecords(records, label){
    const filePath = path.join(DATA_DIR, "events_" + label + ".json");
    fs.writeFileSync(filePath, JSON.stringify(records, null, 2), "utf-8");
    console.log("Wrote [" + records.length + "] to " + filePath);
}

async function runConcurrently(tasks, limit){
    const results = new Array(tasks.length);
    let nextIndex = 0;

    async function worker(){
        while(nextIndex < tasks.length){
            const idx = nextIndex++;
            results[idx] = await tasks[idx]();
        }
    }

    let workers = [];
    for(let i = 0; i < Math.min(limit, tasks.length); i++){
        workers.push(worker());
    }
    
    await Promise.all(workers);
    return results;
}

async function main(){
    // create data dir if does not exist
    if(!fs.existsSync(DATA_DIR)){
        fs.mkdirSync(DATA_DIR, {recursive: true})
    }

    let ranges = generateYearRanges();

    const tasks = ranges.map((range) => {
        return async () => {
            console.log("Starting download for range [ " + range.from + ", " + range.to + ", " + range.label + " ]");
            try{
                let records = await downloadDateRange(range.from, range.to, range.label);
                if(records.length > 0){
                    saveRecords(records, range.label);
                } else {
                    console.log("["+range.label+"] No records to save");
                }
                return {label: range.label, count: records.length, error:null};
            } catch(err){
                return {label: range.label, count: 0, error: err.message};
            }
        }
    });

    const results = await runConcurrently(tasks, CONCURRENCY_LIMIT);

    let totalRecords = 0;
    let failedYears = [];

    for(const result of results){
        totalRecords += result.count;
        if(result.error){
            failedYears.push(result.label);
            console.log("" + result.label +": Failed [" + result.error + "]");
        } else {
            console.log("" + result.label +": " + result.count + " records");
        }
    }

    console.log("-------------------------------------------------------------");
    console.log("\tTotal records downloaded: " + totalRecords);
    console.log("\tFailed year ranges: " + (failedYears.length > 0 ? failedYears.join(", ") : "none"));
    console.log("-------------------------------------------------------------");

    if(failedYears.length > 0){
        console.warn("Some year ranges have failed, please re-run the script to retry.\nAlready downloaded years will be overwritten.");
    }

    console.log("All data downloaded successfully!")
}

main().catch((err) => {
    console.log("Fatal Errror: ", err);
    process.exit(1);
});
