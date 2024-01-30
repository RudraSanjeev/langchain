import readline from "readline";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const greet = async (name) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log(`Good morning ${name}`);
      resolve();
    }, 5000);
  });
};

const askName = async () => {
  while (true) {
    const answer = await new Promise((resolve) => {
      rl.question("Enter a name (or press Ctrl + C to exit): ", resolve);
    });

    await greet(answer);
  }
};

askName();
