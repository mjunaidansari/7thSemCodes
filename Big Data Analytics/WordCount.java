import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.StringTokenizer;

public class WordCount {

	public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		@Override
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String line = value.toString();
			StringTokenizer itr = new StringTokenizer(line);
			while(itr.hasMoreTokens()) {
				word.set(itr.nextToken());
				context.write(word, one);
			}
		}
	}

	public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		private IntWritable result = new IntWritable();

		@Override
		protected void reduce (Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable val: values) {
				sum += val.get();
			}
			result.set(sum);
			context.write(key, result);
		}
	}

	public static void main(String args[]) throws Exception {
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "word count");
		job.setJarByClass(WordCount.class);
		job.setMapperClass(TokenizerMapper.class);
		job.setReducerClass(IntSumReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
/*
 * Steps to Compile and Run the Hadoop MapReduce Program:
 *
 * 1. **Compile the Java Code:**
 *    Ensure Hadoop is installed and configured.
 *    Compile the Java files using Hadoop classpath.
 *    Command:
 *      javac -classpath `hadoop classpath` -d . YourProgram.java
 *
 * 2. **Package the Classes into a JAR File:**
 *    Create a JAR file containing the compiled classes.
 *    Command:
 *      jar -cvf your-program.jar .
 *
 * 3. **Upload Input Data to HDFS:**
 *    Upload the input data file to HDFS.
 *    Command:
 *      hadoop fs -mkdir /input
 *      hadoop fs -put local_input_file.txt /input/
 *
 * 4. **Run the MapReduce Job:**
 *    Execute the MapReduce job using the JAR file and specify input and output paths.
 *    Command:
 *      hadoop jar your-program.jar YourMainClass /input/local_input_file.txt /output
 *
 * 5. **Check the Output:**
 *    View the results of the MapReduce job from the output directory.
 *    Command:
 *      hadoop fs -cat /output/part-r-00000
 *
 */
