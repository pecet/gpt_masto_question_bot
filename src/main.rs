use std::{error::Error, borrow::Cow, collections::HashMap};
use async_openai::{types::CreateCompletionRequestArgs, Client};
use serde::Deserialize;
use reqwest;
use std::env;

#[derive(Clone, Deserialize, Debug)]
struct GptResponse {
    question: String, 
    answers: Vec<String>,
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
        ("poll[expires_in]", "28800".to_owned()),
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
    let gpt_response_string = query_gpt().await?;
    println!("Got response from mighty gpt:");
    println!("{}", &gpt_response_string);
    let gpt_response: GptResponse = serde_json::from_str(&gpt_response_string)?;
    println!("Successfully parsed JSON from gpt:");
    println!("{:#?}", &gpt_response);

    let response = send_mastodon_poll(gpt_response).await?;
    println!("Response from mastodon server:");
    println!("{}", response);   

    Ok(())
}