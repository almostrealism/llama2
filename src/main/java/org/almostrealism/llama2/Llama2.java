package org.almostrealism.llama2;

import io.almostrealism.compute.ComputeRequirement;
import io.almostrealism.profile.OperationProfile;
import org.almostrealism.collect.PackedCollection;
import org.almostrealism.ml.AttentionFeatures;
import org.almostrealism.ml.AutoregressiveModel;
import org.almostrealism.ml.BPE;
import org.almostrealism.model.Model;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.function.Consumer;

public class Llama2 implements AttentionFeatures {
	static {
		System.setProperty("AR_HARDWARE_OFF_HEAP_SIZE", "0");
		System.setProperty("AR_EXPRESSION_WARNINGS", "disabled");
		System.setProperty("AR_GRAPH_PROPAGATION_WARNINGS", "disabled");
	}

	private Config config;
	private Weights weights;
	private String[] vocab;
	private float[] vocabScores;

	private AutoregressiveModel model;
	private OperationProfile profile;

	public static void main(String args[]) throws IOException {
		int steps = 256;

		String checkpoint = args.length > 0 ? args[0] : "stories110M.bin";
		String prompt = args.length > 1 ? args[1] : null;

		Llama2 llama = new Llama2(checkpoint);
		llama.setTemperature(0.0);

		long duration = llama.run(steps, prompt,
				token -> {
					System.out.printf("%s", token);
					System.out.flush();
				});

		System.out.printf("\ntokens per second: %f\n", (steps - 1) / (double) duration * 1000);
		llama.getProfile().print();

		System.out.println("Done");
	}

	public Llama2(String checkpoint) throws IOException {
		long start = System.currentTimeMillis();

		try (FileChannel fileChannel = FileChannel.open(Paths.get(checkpoint), StandardOpenOption.READ)) {
			ByteBuffer bb = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
			bb.order(ByteOrder.LITTLE_ENDIAN);

			// Read header
			config = new Config(bb);
			weights = new Weights(config, bb.asFloatBuffer());
			System.out.println("Loaded weights in " + (System.currentTimeMillis() - start) + "ms");
		}

		// Load the tokenizer
		vocab = new String[config.vocabSize];
		vocabScores = new float[config.vocabSize];
		
		try (FileChannel channel = FileChannel.open(Paths.get("tokenizer.bin"), StandardOpenOption.READ)) {
			ByteBuffer bb = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
			bb.order(ByteOrder.LITTLE_ENDIAN);
			
			int maxTokenLength = bb.getInt();
			for (int i = 0; i < config.vocabSize; i++) {
				vocabScores[i] = bb.getFloat();
				int len = bb.getInt();
				byte[] bytes = new byte[len];
				bb.get(bytes);
				vocab[i] = new String(bytes);
			}
		}

		profile = new OperationProfile();
		model = model(profile);
	}

	public OperationProfile getProfile() { return profile; }

	public void setTemperature(double temperature) {
		model.setTemperature(temperature);
	}

	protected AutoregressiveModel model(OperationProfile profile, ComputeRequirement... requirements) {
		Model transformer = new Model(shape(config.dim));

		// Placeholder for the index of the current step
		PackedCollection<?> position = new PackedCollection<>(1);

		int dim = config.dim;

		// Transformer layers
		for (int i = 0; i < config.layerCount; i++) {
			transformer.add(transformer(config.headCount,
					weights.rmsAttWeights.range(shape(config.dim), i * dim),
					weights.wk.range(shape(dim, dim), dim * dim * i),
					weights.wv.range(shape(dim, dim), dim * dim * i),
					weights.wq.range(shape(dim, dim), dim * dim * i),
					weights.wo.range(shape(dim, dim), dim * dim * i),
					weights.freqCis,
					weights.rmsFfn.range(shape(config.dim), i * config.dim),
					weights.w1.range(shape(config.hiddenDim, dim), dim * config.hiddenDim * i),
					weights.w2.range(shape(dim, config.hiddenDim), dim * config.hiddenDim * i),
					weights.w3.range(shape(config.hiddenDim, dim), dim * config.hiddenDim * i),
					p(position), requirements));
		}

		// Final RMS Norm
		transformer.add(rmsnorm(weights.rmsFinalWeight));

		// Logits
		transformer.add(dense(weights.wcls));

		return AutoregressiveModel.of(transformer.compile(false, profile),
				step -> position.setMem((double) step),
				t -> weights.tokenEmbeddings.range(shape(config.dim), t * config.dim));
	}

	public long run(int steps, String prompt, Consumer<String> output) {
		if (steps <= 0 || steps > config.seqLen) {
			steps = config.seqLen;
		}

		// Prepare the prompt
		int[] promptTokens = null;
		int promptTokenCount = 0;
		if (prompt != null) {
			promptTokens = new int[config.seqLen];
			promptTokenCount = BPE.encode(prompt, vocab, vocabScores, config.vocabSize, promptTokens);
		}

		long start = 0;
		int next;
		int token = 1;

		model.setCurrentToken(1);
		model.setPrompt(promptTokens, promptTokenCount);

		output.accept("<s>\n");
		while (model.getCurrentStep() < steps) {
			next = model.next();

			// Strip leading whitespace following token 1
			String tokenStr = (token == 1 && vocab[next].charAt(0) == ' ') ? vocab[next].substring(1) : vocab[next];
			output.accept(tokenStr);
			token = next;

			if (start == 0) {
				start = System.currentTimeMillis();
			}
		}

		return System.currentTimeMillis() - start;
	}
}
