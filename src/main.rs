use std::{error::Error};
use async_openai::{types::CreateCompletionRequestArgs, Client};
use serde::{Deserialize, Serialize};
use reqwest;
use std::env;
use std::fs;
use str_distance::*;

#[derive(Clone, Deserialize, Serialize, Debug)]
struct GptResponse {
    question: String,
    answers: Vec<String>,
}

impl GptResponse {
    fn check_anwsers_length(&self) -> bool {
        for answer in &self.answers {
            if answer.len() > 50 {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Deserialize, Serialize, Debug, Default)]
struct PreviousGptResponses {
    responses: Vec<GptResponse>,
}

#[derive(Clone, Deserialize, Debug)]
struct Similarity {
    similarity: f64,
}

impl PreviousGptResponses {
    fn push(&mut self, value: GptResponse) {
        self.responses.push(value);
    }

    fn normalize_string(&self, input: &String) -> String {
        let output = input.to_ascii_lowercase();
        let what_to_replace = vec!["what", "where", "who", "which", "do you", "whom", "consider", "opinion", "?", "think", "you", "to have"];
        for from in what_to_replace {
            output.replace(from, "");
        }
        output
    }

    fn compute_similarlity_of_all(&self, question_to_compare: &String) -> f64 {
        let mut similarlity = 0.0_f64;
        for response in &self.responses {
            let first_string = self.normalize_string(&question_to_compare);
            let second_string = self.normalize_string(&response.question);
            let value = 1.0_f64 - str_distance_normalized(first_string, second_string, DamerauLevenshtein::default());
            if value > similarlity {
                similarlity = value;
            }
        }
        similarlity
    }

    async fn query_similarlity(&self, question_to_compare: &String, take_last: i64) -> Result<f64, Box<dyn Error>> {
        let v = &self.responses;
        let lower_bound = v.len() as i64 - take_last;
        let lower_bound = if lower_bound < 0_i64 {
            0_i64
        } else {
            lower_bound
        } as usize;
        let last_reponses = &v[lower_bound .. v.len()];
        if last_reponses.len() > 0 {
            let client = Client::new();
            let prompt_intro = r#"Respond only with JSON with float field "similarity". Compare similarity of first sentence with other sentences, return max value."#;
            let mut prompt_items = "".to_owned();
            prompt_items += &format!("Sentence 1: \"{}\"\n", question_to_compare)[..];
            for (i, response) in last_reponses.iter().enumerate() {
                prompt_items += &format!("Sentence {}: \"{}\"\n", i + 2, response.question)[..];
            }
            let prompt = format!("{}\n{}", prompt_intro, prompt_items);
            println!("Prompt to query last sentences:\n{}", &prompt);
            let request = CreateCompletionRequestArgs::default()
                .model("text-davinci-003")
                .prompt(prompt)
                .max_tokens(512_u16)
                .temperature(0.1)
                .frequency_penalty(0.0)
                .presence_penalty(0.0)
                .build()?;
            let response = client.completions().create(request).await?;
            let first_response = response.choices.get(0).ok_or("No first item in response")?;
            let gpt_response: Similarity = serde_json::from_str(&first_response.text.to_owned())?;
            Ok(gpt_response.similarity)
        } else {
            Ok(0.0)
        }
    }
}

async fn query_gpt() -> Result<String, Box<dyn Error>> {
    let client = Client::new();
    let prompt = r#"Respond only with JSON containing field "question" and array "answers". Question should be random question about opinion. There should be 4 answers, each answer should be 35 chars max, last should be funny."#;
    let request = CreateCompletionRequestArgs::default()
        .model("text-davinci-003")
        .prompt(prompt)
        .max_tokens(256_u16)
        .temperature(0.99)
        .frequency_penalty(0.0)
        .presence_penalty(1.8)
        .build()?;
    let response = client.completions().create(request).await?;
    let first_response = response.choices.get(0).ok_or("No first item in response")?;
    Ok(first_response.text.to_owned())
}

async fn send_mastodon_poll(gpt_response: GptResponse) -> Result<String, Box<dyn Error>> {
    let params = [
        ("status", gpt_response.question.clone()),
        ("visibility", "public".to_owned()),
        ("language", "en".to_owned()),
        ("poll[options][]", gpt_response.answers[0].clone()),
        ("poll[options][]", gpt_response.answers[1].clone()),
        ("poll[options][]", gpt_response.answers[2].clone()),
        ("poll[options][]", gpt_response.answers[3].clone()),
        ("poll[expires_in]", "72000".to_owned()),
    ];
    let instance = env::var("MAST_INSTANCE")?;
    let token = env::var("MAST_TOKEN")?;
    let url = format!("https://{instance}/api/v1/statuses");
    let client = reqwest::Client::new();
    let response = client
        .post(url)
        .bearer_auth(token)
        .form(&params)
        .send()
        .await?;
    let text = response.text().await?;
    Ok(text)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "responses.txt";
    let previous_responses_string = fs::read_to_string(file_path).unwrap_or(String::new());
    let mut previous_responses: PreviousGptResponses = serde_json::from_str(&previous_responses_string).unwrap_or(PreviousGptResponses::default());
    println!("Loaded {} previous responses", &previous_responses.responses.len());

    let retries = 8;

    for i in 1..retries {
        println!("---- Going for GPT response retry: {} of {}", &i, &retries);
        let gpt_response_string = query_gpt().await?;
        println!("Got response from mighty gpt:");
        println!("{}", &gpt_response_string);
        let gpt_response: GptResponse = serde_json::from_str(&gpt_response_string)?;
        println!("Successfully parsed JSON from gpt:");
        println!("{:#?}", &gpt_response);

        let computed_similarlity = previous_responses.compute_similarlity_of_all(&gpt_response.question.to_owned());
        println!("Similarlity computer locally: {}", computed_similarlity);
    
        if gpt_response.check_anwsers_length() && computed_similarlity <= 0.49_f64{
            let similarlity = previous_responses.query_similarlity(&gpt_response.question, 8).await?;
            println!("Similarlity from GPT: {}", &similarlity);
            if similarlity <= 0.35_f64 {
                // println!("**** Using response as similarity <= 0.35 ***");
                // let response = send_mastodon_poll(gpt_response.clone()).await?;
                // println!("Response from mastodon server:");
                // println!("{}", response);

                // save response
                previous_responses.push(gpt_response.clone());
                let previous_responses_string = serde_json::to_string_pretty(&previous_responses)?;
                fs::write(file_path, previous_responses_string)?;
                println!("Response saved");
                break;
            }
        }
    }

    Ok(())
}