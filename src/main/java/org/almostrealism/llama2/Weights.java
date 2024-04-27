package org.almostrealism.llama2;

import io.almostrealism.collect.TraversalPolicy;
import org.almostrealism.CodeFeatures;
import org.almostrealism.collect.PackedCollection;

import java.nio.FloatBuffer;

public class Weights implements CodeFeatures {
	// token embedding table
	public final PackedCollection<?> tokenEmbeddings; // (vocab_size, dim)

	// weights for rmsnorms
	public final PackedCollection<?> rmsAttWeights; // (layer, dim)

	// weights for matmuls
	public final PackedCollection<?> wq; // (layer, dim, dim)
	public final PackedCollection<?> wk; // (layer, dim, dim)
	public final PackedCollection<?> wv; // (layer, dim, dim)
	public final PackedCollection<?> wo; // (layer, dim, dim)
	public PackedCollection<?> rmsFfn; // (layer, dim)

	// weights for ffn
	public final PackedCollection<?> w1; // (layer, hidden_dim, dim)
	public final PackedCollection<?> w2; // (layer, dim, hidden_dim)
	public final PackedCollection<?> w3; // (layer, hidden_dim, dim)

	// final rmsnorm
	public final PackedCollection<?> rmsFinalWeight; // (dim,)

	// freq cis for RoPE relatively positional embeddings
	public final PackedCollection<?> freqCis;

	// classifier weights for the logits, on the last layer
	public final PackedCollection<?> wcls;

	public Weights(Config config, FloatBuffer buffer) {
		this.tokenEmbeddings =
				pack(take(buffer, config.vocabSize, config.dim))
				.reshape(shape(config.vocabSize, config.dim));

		this.rmsAttWeights =
				pack(take(buffer, config.layerCount, config.dim))
				.reshape(shape(config.layerCount, config.dim));

		this.wq = pack(take(buffer, config.layerCount, config.dim, config.dim))
				.reshape(shape(config.layerCount, config.dim, config.dim));
		this.wk = pack(take(buffer, config.layerCount, config.dim, config.dim))
				.reshape(shape(config.layerCount, config.dim, config.dim));
		this.wv = pack(take(buffer, config.layerCount, config.dim, config.dim))
				.reshape(shape(config.layerCount, config.dim, config.dim));
		this.wo = pack(take(buffer, config.layerCount, config.dim, config.dim))
				.reshape(shape(config.layerCount, config.dim, config.dim));

		this.rmsFfn = pack(take(buffer, config.layerCount, config.dim))
				.reshape(shape(config.layerCount, config.dim));

		this.w1 = pack(take(buffer, config.layerCount, config.hiddenDim, config.dim))
				.reshape(shape(config.layerCount, config.hiddenDim, config.dim));
		this.w2 = pack(take(buffer, config.layerCount, config.dim, config.hiddenDim))
				.reshape(shape(config.layerCount, config.dim, config.hiddenDim));
		this.w3 = pack(take(buffer, config.layerCount, config.hiddenDim, config.dim))
				.reshape(shape(config.layerCount, config.hiddenDim, config.dim));

		this.rmsFinalWeight =
				pack(take(buffer, config.dim))
				.reshape(shape(config.dim));

		this.freqCis = packComplex(
				take(buffer, config.seqLen, config.headSize / 2),
				take(buffer, config.seqLen, config.headSize / 2),
				shape(config.seqLen, config.headSize / 2, 2));

		this.wcls = config.sharedWeights ? tokenEmbeddings : null;
	}

	static float[] take(FloatBuffer buffer, int... dims) {
		TraversalPolicy shape = new TraversalPolicy(dims);
		float[] floats = new float[shape.getTotalSize()];
		buffer.get(floats);
		return floats;
	}

	static PackedCollection<?> packComplex(float real[], float imag[], TraversalPolicy shape) {
		if (shape.length(shape.getDimensions() - 1) != 2)
			throw new IllegalArgumentException();

		double data[] = new double[shape.getTotalSize()];
		for (int i = 0; i < data.length; i += 2) {
			data[i] = real[i / 2];
			data[i + 1] = imag[i / 2];
		}

		PackedCollection<?> c = new PackedCollection<>(shape);
		c.setMem(0, data, 0, shape.getTotalSize());
		return c;
	}
}
