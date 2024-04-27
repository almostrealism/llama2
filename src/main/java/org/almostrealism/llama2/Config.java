package org.almostrealism.llama2;

import java.nio.ByteBuffer;

public class Config {
	/** transformer dimension */
	public final int dim;

	/** for ffn layers */
	public final int hiddenDim;

	/** number of layers */
	public final int layerCount;

	/** number of query heads */
	public final int headCount;

	/** number of key/value heads */
	public final int kvHeadCount;

	/** vocabulary size, usually 256 (byte-level) */
	public final int vocabSize;

	/** sequence length */
	public final int seqLen;

	/** whether to share weights between layers */
	public final boolean sharedWeights;

	/** size of each head */
	public final int headSize;

	public Config(ByteBuffer buffer) {
		this.dim = buffer.getInt();
		this.hiddenDim = buffer.getInt();
		this.layerCount = buffer.getInt();
		this.headCount = buffer.getInt();
		this.kvHeadCount = buffer.getInt();

		int vocabSize = buffer.getInt();
		this.vocabSize = Math.abs(vocabSize);
		this.seqLen = buffer.getInt();
		this.sharedWeights = vocabSize > 0;
		this.headSize = dim / headCount;
	}
}
